import argparse
import json
import os
import random
import time
from itertools import product, islice

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from tqdm import tqdm

import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer

from grad_cache.functional import cached, cat_input_tensor

class Config:
    def __init__(self):
        self.datasets = ["dbpedia-entity", "climate-fever", "msmarco"]

        self.dev_dataset_sizes = {
            "dbpedia-entity": 54367,
            "climate-fever": 53140,
            "msmarco": 77856,
        }

        self.train_file_paths = {
            "dbpedia-entity": "../reranker/beir.all.dbpedia-entity.train.generated_queries.listwise.jsonl",
            "climate-fever":  "../reranker/beir.all.climate-fever.train.generated_queries.listwise.jsonl",
            "msmarco":        "../reranker/beir.all.msmarco.train.generated_queries.listwise.jsonl",
        }
        self.dev_file_paths = {
            "dbpedia-entity": "../reranker/beir.all.dbpedia-entity.dev.generated_queries.listwise.jsonl",
            "climate-fever":  "../reranker/beir.all.climate-fever.dev.generated_queries.listwise.jsonl",
            "msmarco":        "../reranker/beir.all.msmarco.dev.generated_queries.listwise.jsonl",
        }

        self.retriever_k = 20
        self.list_length = 20
        self.accumulation_steps = 256
        self.temp = 0.01
        self.dropout = 0.00
        self.batch_size = 16
        self.lr = 2e-4
        self.rank_temp = 0.05
        self.kl_temp = 0.3
        self.contrastive_loss_weight = 0.10
        self.instruction = "query: "
        self.query_maxlength = 64
        self.text_maxlength = 512
        self.num_epochs = 30
        self.weight_decay = 1e-2
        self.model_name_or_path = 'intfloat/e5-base-unsupervised'
        self.save_model = True
        self.threshold_score = 0.6
        self.save_model_name = f"multi_dataset_e5_embedding_model"


class EmbeddingModel(nn.Module):
    def __init__(self, model_name_or_path, dropout=0.0):
        super(EmbeddingModel, self).__init__()

        configuration = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        configuration.hidden_dropout_prob = dropout
        configuration.attention_probs_dropout_prob = dropout

        self.bert = AutoModel.from_pretrained(
            model_name_or_path, config=configuration, trust_remote_code=True
        )

    def forward(self, ids, mask):
        outputs = self.bert(ids, mask)
        pooled_output = self.mean_pooling(outputs, mask)
        return torch.nn.functional.normalize(pooled_output, p=2, dim=1)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )


@cached
@torch.amp.autocast('cuda', dtype=torch.bfloat16)
def call_model(model, ids, mask):
    return model(ids, mask)


class RankingIterableDataset(IterableDataset):
    """
    Streams dev file line by line, no shuffling.
    """
    def __init__(self, file_path, list_length):
        super().__init__()
        self.file_path = file_path
        self.list_length = list_length

    def __iter__(self):
        worker_info = get_worker_info()
        # Open the file
        with open(self.file_path, 'r', encoding='utf-8') as f:
            # If using multiple workers, partition the file lines:
            if worker_info is not None:
                f = islice(f, worker_info.id, None, worker_info.num_workers)
            # Iterate through the (possibly partitioned) file
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                query = item["query"]
                passages = item["passages"][: self.list_length]
                texts = ["passage: " + p["text"] for p in passages]
                ids = [p["docid"] for p in passages]
                scores = [p["score"] for p in passages]
                yield (query, texts, ids, scores)


class RankingShuffledIterableDataset(IterableDataset):
    """
    Performs local shuffling using a buffer of size `shuffle_buffer_size`.
    Once the buffer fills, shuffle it, yield it, then refill, etc.
    """
    def __init__(self, file_path, list_length, shuffle_buffer_size=262144):
        super().__init__()
        self.file_path = file_path
        self.list_length = list_length
        self.shuffle_buffer_size = shuffle_buffer_size

    def __iter__(self):
        worker_info = get_worker_info()
        with open(self.file_path, 'r', encoding='utf-8') as f:
            if worker_info is not None:
                f = islice(f, worker_info.id, None, worker_info.num_workers)
            buffer = []
            for line in f:
                line = line.strip()
                if not line:
                    continue

                item = json.loads(line)
                query = item["query"]
                passages = item["passages"][: self.list_length]
                texts = ["passage: " + p["text"] for p in passages]
                ids = [p["docid"] for p in passages]
                scores = [p["score"] for p in passages]
                buffer.append((query, texts, ids, scores))

                # If the buffer is full, shuffle and yield
                if len(buffer) >= self.shuffle_buffer_size:
                    random.shuffle(buffer)
                    for x in buffer:
                        yield x
                    buffer = []

            # Yield any remaining items
            if buffer:
                random.shuffle(buffer)
                for x in buffer:
                    yield x


# -------------------------------------------------------------------------
# Collator function
# -------------------------------------------------------------------------
class Collator(object):
    def __init__(self, tokenizer, instruction, query_maxlength, text_maxlength, list_length):
        self.tokenizer = tokenizer
        self.instruction = instruction
        self.query_maxlength = query_maxlength
        self.text_maxlength = text_maxlength
        self.list_length = list_length

    def __call__(self, batch):
        """
        batch is a list of N items: (query, list_of_texts, list_of_ids, list_of_scores)
        Produces tokenized query and passages.
        """
        query_texts = [self.instruction + example[0] for example in batch]

        all_passage_texts = []
        all_passage_ids = []
        all_passage_scores = []

        for example in batch:
            texts_20 = example[1]
            ids_20 = example[2]
            scores_20 = example[3]
            all_passage_texts.extend(texts_20)
            all_passage_ids.extend(ids_20)
            all_passage_scores.extend(scores_20)

        p_queries = self.tokenizer.batch_encode_plus(
            query_texts,
            max_length=self.query_maxlength,
            padding=True,
            return_tensors='pt',
            truncation=True,
        )

        p_all_passages = self.tokenizer.batch_encode_plus(
            all_passage_texts,
            max_length=self.text_maxlength,
            padding=True,
            return_tensors='pt',
            truncation=True,
        )

        return (
            p_queries['input_ids'],
            p_queries['attention_mask'],
            p_all_passages['input_ids'],
            p_all_passages['attention_mask'],
            all_passage_ids,
            all_passage_scores,
        )


# -------------------------------------------------------------------------
# Functions to get DataLoaders
# -------------------------------------------------------------------------
def get_train_dataloader(dset, config, tokenizer, shuffle_buffer_size=262144):
    ds = RankingShuffledIterableDataset(
        file_path=config.train_file_paths[dset],
        list_length=config.list_length,
        shuffle_buffer_size=shuffle_buffer_size
    )
    dataloader = DataLoader(
        ds,
        batch_size=config.batch_size,
        collate_fn=Collator(
            tokenizer,
            config.instruction,
            config.query_maxlength,
            config.text_maxlength,
            config.list_length
        ),
        num_workers=4,
        drop_last=True,
        pin_memory=False
    )
    return dataloader

def get_dev_dataloader(dset, config, tokenizer):
    ds = RankingIterableDataset(
        file_path=config.dev_file_paths[dset],
        list_length=config.list_length
    )
    dataloader = DataLoader(
        ds,
        batch_size=config.batch_size,
        collate_fn=Collator(
            tokenizer,
            config.instruction,
            config.query_maxlength,
            config.text_maxlength,
            config.list_length
        ),
        num_workers=4,
        drop_last=True,
        pin_memory=False
    )
    return dataloader


# -------------------------------------------------------------------------
# Trainer
# -------------------------------------------------------------------------
class Trainer:
    def __init__(self, model, optimizer, scheduler, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

        self.best_overall_dev_loss = float('inf')
        self.best_rank_dev_loss = float('inf')
        self.non_improvement_count = 0

        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

    @cat_input_tensor
    def rank_loss(self, query_embeddings, all_passage_embeddings, all_passage_scores):
        """
        Listwise KL divergence:
        Compare softmax of dot-products to softmax of ground-truth scores
        """
        bs = len(query_embeddings)
        all_passage_embeddings = all_passage_embeddings.reshape(bs, self.config.list_length, -1)
        y_preds = torch.bmm(all_passage_embeddings, query_embeddings.unsqueeze(-1)).squeeze(-1).double()

        all_passage_scores = torch.tensor(all_passage_scores).cuda().reshape(bs, self.config.list_length)
        rank_temp = torch.tensor(self.config.rank_temp).double().cuda()
        kl_temp = torch.tensor(self.config.kl_temp).double().cuda()

        y_preds = F.log_softmax(y_preds / rank_temp, dim=-1)
        all_passage_scores = F.log_softmax(all_passage_scores / kl_temp, dim=-1)

        loss = self.kl_loss(y_preds, all_passage_scores).float()
        return loss

    @cat_input_tensor
    def contrastive_loss(self, query_embeddings, all_passage_embeddings, all_ids, all_passage_scores):
        """
        Contrastive loss with positive (top passage) vs. (hard) negatives.
        """
        num_queries = len(query_embeddings)
        assert len(all_passage_embeddings) == num_queries * self.config.list_length

        pos_passage_embeddings = []
        hard_negative_passage_embeddings = []
        pos_passage_index_map = {}
        hn_passage_index_map = {}

        pos_idx = 0
        hn_idx = 0
        no_contrast_ids = []
        curr_no_contrast_ids = []
        top_passage_score = 0
        rel_passage_scores = []

        for i in range(len(all_passage_embeddings)):
            if i % self.config.list_length == 0:
                top_passage_score = all_passage_scores[i]
                rel_passage_scores.append(top_passage_score)
                pos_passage_embeddings.append(all_passage_embeddings[i])
                if all_ids[i] not in pos_passage_index_map:
                    pos_passage_index_map[all_ids[i]] = []
                pos_passage_index_map[all_ids[i]].append(pos_idx)
                pos_idx += 1

                if i != 0:
                    no_contrast_ids.append(curr_no_contrast_ids)
                curr_no_contrast_ids = [all_ids[i]]
            else:
                score = all_passage_scores[i]
                hard_negative_passage_embeddings.append(all_passage_embeddings[i])
                if all_ids[i] not in hn_passage_index_map:
                    hn_passage_index_map[all_ids[i]] = []
                hn_passage_index_map[all_ids[i]].append(hn_idx)
                hn_idx += 1

                if score > self.config.threshold_score * top_passage_score:
                    curr_no_contrast_ids.append(all_ids[i])

        if len(curr_no_contrast_ids) != 0:
            no_contrast_ids.append(curr_no_contrast_ids)

        pos_passage_embeddings = torch.stack(pos_passage_embeddings)
        rel_passage_scores = torch.tensor(rel_passage_scores).cuda()
        assert (len(rel_passage_scores) == len(pos_passage_embeddings))
        hard_negative_passage_embeddings = torch.stack(hard_negative_passage_embeddings)

        # Build "no-contrast" masks
        pos_not_contrast_mask = torch.ones(len(pos_passage_embeddings), num_queries).cuda()
        for i, doc_ids in enumerate(no_contrast_ids):
            for doc_id in doc_ids:
                if doc_id in pos_passage_index_map:
                    for idx_ in pos_passage_index_map[doc_id]:
                        if idx_ != i:
                            pos_not_contrast_mask[idx_, i] = 0

        hn_not_contrast_mask = torch.ones(len(hard_negative_passage_embeddings), num_queries).cuda()
        for i, doc_ids in enumerate(no_contrast_ids):
            for doc_id in doc_ids:
                if doc_id in hn_passage_index_map:
                    for idx_ in hn_passage_index_map[doc_id]:
                        hn_not_contrast_mask[idx_, i] = 0

        temp = torch.tensor(self.config.temp).double().cuda()
        pos_query_scores = torch.matmul(pos_passage_embeddings, query_embeddings.T).double()
        pos_query_scores = torch.exp(pos_query_scores / temp) * pos_not_contrast_mask

        hn_query_scores = torch.matmul(hard_negative_passage_embeddings, query_embeddings.T).double()
        hn_query_scores = torch.exp(hn_query_scores / temp) * hn_not_contrast_mask

        losses = torch.diagonal(pos_query_scores) / (pos_query_scores.sum(dim=0) + hn_query_scores.sum(dim=0))
        losses = torch.log(losses).float()
        return -losses.mean()

    # ---------------------------------------------------------------------
    # Perform partial training (accum steps)
    # ---------------------------------------------------------------------
    def partial_train(self, data_iter, max_accum_steps=512):
        """
        Reads up to `max_accum_steps` mini-batches from data_iter.
        Accumulates grads, does one optimizer step.
        Returns how many batches were consumed (could be < max_accum_steps if data ended).
        """
        self.model.train()

        cache_q_emb = []
        cache_p_emb = []
        cache_ids = []
        cache_scores = []
        closures_q = []
        closures_p = []

        steps_done = 0

        for _ in range(max_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                # no more data
                break

            (
                q_ids,
                q_mask,
                p_ids,
                p_mask,
                docids,
                scores
            ) = batch

            # Forward calls
            q_emb, f_q = call_model(self.model, q_ids.cuda(), q_mask.cuda())
            p_emb, f_p = call_model(self.model, p_ids.cuda(), p_mask.cuda())

            cache_q_emb.append(q_emb)
            cache_p_emb.append(p_emb)
            cache_ids.extend(docids)
            cache_scores.extend(scores)

            closures_q.append(f_q)
            closures_p.append(f_p)

            steps_done += 1


        if steps_done == max_accum_steps:
            c_loss = self.contrastive_loss(cache_q_emb, cache_p_emb, cache_ids, cache_scores)
            r_loss = self.rank_loss(cache_q_emb, cache_p_emb, cache_scores)
            loss = c_loss * self.config.contrastive_loss_weight + r_loss

            loss.backward()

            # Apply closures
            for fn, data_emb in zip(closures_q, cache_q_emb):
                fn(data_emb)
            for fn, data_emb in zip(closures_p, cache_p_emb):
                fn(data_emb)

            # Gradient step
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return steps_done

    # ---------------------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------------------
    def evaluate(self, dev_dataloader, total_batches):
        self.model.eval()
        cache_q_emb = []
        cache_p_emb = []
        cache_ids = []
        cache_scores = []
        total_c_loss = 0.0
        total_r_loss = 0.0
        num_accum = 0

        data_iter = iter(dev_dataloader)
        pbar = tqdm(total=total_batches, desc="Evaluating", leave=False)
        step = 0
        with torch.no_grad():
            while True:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                step += 1

                (
                    q_ids,
                    q_mask,
                    p_ids,
                    p_mask,
                    docids,
                    scores
                ) = batch

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    q_emb = self.model(q_ids.cuda(), q_mask.cuda())
                    p_emb = self.model(p_ids.cuda(), p_mask.cuda())

                cache_q_emb.append(q_emb)
                cache_p_emb.append(p_emb)
                cache_ids.extend(docids)
                cache_scores.extend(scores)

                # Evaluate in chunks of accumulation_steps
                if (step % self.config.accumulation_steps) == 0:
                    num_accum += 1
                    c_val = self.contrastive_loss(cache_q_emb, cache_p_emb, cache_ids, cache_scores)
                    r_val = self.rank_loss(cache_q_emb, cache_p_emb, cache_scores)
                    total_c_loss += c_val.item()
                    total_r_loss += r_val.item()

                    # Clear caches
                    cache_q_emb = []
                    cache_p_emb = []
                    cache_ids = []
                    cache_scores = []
                    pbar.update(1)

            # Leftover in cache, but no loss calculated yet
            if len(cache_q_emb) > 0 and step < self.config.accumulation_steps:
                num_accum += 1
                c_val = self.contrastive_loss(cache_q_emb, cache_p_emb, cache_ids, cache_scores)
                r_val = self.rank_loss(cache_q_emb, cache_p_emb, cache_scores)
                total_c_loss += c_val.item()
                total_r_loss += r_val.item()

        if num_accum == 0:
            # Edge case if dev is empty or very small
            return 0.0, 0.0

        avg_c_loss = total_c_loss / num_accum
        avg_r_loss = total_r_loss / num_accum

        print("CONTRASTIVE DEV LOSS:", avg_c_loss)
        print("LISTWISE DEV LOSS:", avg_r_loss)

        return avg_c_loss, avg_r_loss


# -------------------------------------------------------------------------
# Simple warmup+decay scheduler
# -------------------------------------------------------------------------
def create_scheduler(optimizer, total_steps, warmup_steps=20):
    def warmup_then_decay(current_step: int):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(warmup_steps)
        else:
            # Linear decay
            decay_factor = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.1, 1.0 - decay_factor)  # not to go below LR factor 0.1
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_then_decay)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    config = Config()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

    emb_model = EmbeddingModel(config.model_name_or_path, dropout=config.dropout).cuda()
    optimizer = torch.optim.AdamW(emb_model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Estimate total steps in an epoch (approx)
    total_steps_in_epoch = 0
    for dset in config.datasets:
        steps = config.dev_dataset_sizes[dset] * 9 // config.batch_size
        total_steps_in_epoch += steps // config.accumulation_steps

    warmup_steps = 20
    total_steps = config.num_epochs * total_steps_in_epoch

    scheduler = create_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)
    trainer = Trainer(emb_model, optimizer, scheduler, config)

    for epoch in range(config.num_epochs + 1):
        print(f"\n======== EPOCH {epoch}/{config.num_epochs} ========")

        # Evaluate on dev sets (weighted by dataset size)
        total_c_loss = 0.0
        total_r_loss = 0.0
        sum_sizes = sum(config.dev_dataset_sizes[d] for d in config.datasets)

        for dset in config.datasets:
            print(f"\nEvaluating on {dset} dev set ...")
            dev_loader = get_dev_dataloader(dset, config, tokenizer)
            total_batches = config.dev_dataset_sizes[dset] // config.batch_size // config.accumulation_steps
            c_loss, r_loss = trainer.evaluate(dev_loader, total_batches)
            total_c_loss += c_loss * config.dev_dataset_sizes[dset]
            total_r_loss += r_loss * config.dev_dataset_sizes[dset]

            del dev_loader
            torch.cuda.empty_cache()

        avg_contrastive_dev_loss = total_c_loss / sum_sizes
        avg_rank_dev_loss = total_r_loss / sum_sizes
        overall_dev_loss = avg_contrastive_dev_loss * config.contrastive_loss_weight + avg_rank_dev_loss

        print(f"\nWeighted DEV LOSS (Contrastive): {avg_contrastive_dev_loss:.4f}")
        print(f"Weighted DEV LOSS (Rank):        {avg_rank_dev_loss:.4f}")
        print(f"Weighted DEV LOSS (Overall):     {overall_dev_loss:.4f}")

        # Early stopping logic
        trainer.non_improvement_count += 1

        # Check rank dev loss
        if avg_rank_dev_loss < trainer.best_rank_dev_loss:
            trainer.best_rank_dev_loss = avg_rank_dev_loss
            trainer.non_improvement_count = 0

        # Check overall dev loss
        if overall_dev_loss < trainer.best_overall_dev_loss:
            trainer.best_overall_dev_loss = overall_dev_loss
            trainer.non_improvement_count = 0

        if config.save_model:
            emb_model.bert.save_pretrained(config.save_model_name + f'_epoch_{epoch}')
            print(f"Saved model to {config.save_model_name}_epoch_{epoch}")

        if trainer.non_improvement_count >= 2:
            print("Early stopping due to no improvement.")
            break

        if epoch == config.num_epochs:
            break

        # -----------------------------------------------------------------
        # Interleave training across all datasets in this epoch
        # -----------------------------------------------------------------
        train_iters = {}
        # Create a DataLoader & iterator for each dataset
        for dset in config.datasets:
            dl = get_train_dataloader(dset, config, tokenizer, shuffle_buffer_size=262144)  
            train_iters[dset] = iter(dl)

        exhausted = {dset: False for dset in config.datasets}

        pbar = tqdm(total=total_steps_in_epoch, desc="Training Epoch", leave=True)

        # Cycle through datasets until all are exhausted
        while not all(exhausted.values()):
            for dset in config.datasets:
                if exhausted[dset]:
                    continue
                steps_done = trainer.partial_train(train_iters[dset], max_accum_steps=config.accumulation_steps)
                pbar.update(1)
                if steps_done < config.accumulation_steps:
                    # means no more data for dset
                    exhausted[dset] = True

        pbar.close()

    print("Training complete!")


if __name__ == "__main__":
    main()
