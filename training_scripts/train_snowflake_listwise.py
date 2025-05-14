import argparse
import json
import os
import pickle
import random
import time
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer

from grad_cache.functional import cached, cat_input_tensor

class Config:
    def __init__(self, dataset='msmarco'):
        self.dataset = dataset
        self.retriever_k = 20
        self.list_length = 20
        self.accumulation_steps = 256
        self.temp = 0.01
        self.dropout = 0.00
        self.batch_size = 16
        self.lr = 2e-4
        self.rank_temp = 0.05
        self.kl_temp = 0.3
        self.contrastive_loss_weight = 0.1
        self.instruction = "Represent this sentence for searching relevant passages: "
        self.query_maxlength = 64
        self.text_maxlength = 512
        self.num_epochs = 30
        self.weight_decay = 0.01
        self.model_name_or_path = 'Snowflake/snowflake-arctic-embed-m-v1.5'
        self.save_model = True
        self.threshold_score = 0.6
        self.save_model_name = f"{self.dataset}_snowflake_embedding_model"
        self.train_file_path = f'../RankT5/beir.snowflake.{self.dataset}.train.generated_queries.listwise.jsonl'
        self.dev_file_path = f'../RankT5/beir.snowflake.{self.dataset}.dev.generated_queries.listwise.jsonl'
        
class EmbeddingModel(nn.Module):
    def __init__(self, model_name_or_path, dropout=0.0):
        super(EmbeddingModel, self).__init__()

        configuration = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        configuration.hidden_dropout_prob = dropout
        configuration.attention_probs_dropout_prob = dropout

        self.bert = AutoModel.from_pretrained(
            model_name_or_path, config=configuration, trust_remote_code=True
        )
        self.bert.gradient_checkpointing_enable()

    def forward(self, ids, mask):
        outputs = self.bert(ids, mask)
        pooled_output = outputs[0][:, 0]
        return torch.nn.functional.normalize(pooled_output, p=2, dim=1)

@cached
@torch.amp.autocast('cuda', dtype=torch.bfloat16)
def call_model(model, ids, mask):
    return model(ids, mask)

class RankingDataset(Dataset):
    def __init__(self, queries, list_texts, list_ids, list_scores, list_length):
        self.queries = queries
        self.list_texts = list_texts
        self.list_ids = list_ids
        self.list_scores = list_scores
        self.list_length = list_length

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        return (
            self.queries[index],
            self.list_texts[index][:self.list_length],
            self.list_ids[index][:self.list_length],
            self.list_scores[index][:self.list_length],
        )

class Collator(object):
    def __init__(self, tokenizer, instruction, query_maxlength, text_maxlength, list_length):
        self.tokenizer = tokenizer
        self.instruction = instruction
        self.query_maxlength = query_maxlength
        self.text_maxlength = text_maxlength
        self.list_length = list_length

    def __call__(self, batch):
        query_texts = [self.instruction + example[0] for example in batch]
        all_passage_texts = []
        all_passage_ids = []
        all_passage_scores = []

        for example in batch:
            assert len(example[1]) == self.list_length
            assert len(example[2]) == self.list_length
            assert len(example[3]) == self.list_length

            all_passage_texts.extend(example[1])
            all_passage_ids.extend(example[2])
            all_passage_scores.extend(example[3])

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

def load_dataset(file_path, list_length):
    queries = []
    passage_lists_texts = []
    passage_lists_ids = []
    passage_lists_scores = []

    with open(file_path, 'r', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            line = line.strip()
            rankings_dict = json.loads(line)
            queries.append(rankings_dict['query'])
            passages = rankings_dict['passages'][:list_length]
            for passage in passages:
                passage_lists_texts.append(passage['text'])
                passage_lists_ids.append(passage['docid'])
                passage_lists_scores.append(passage['rankt5_score'])

    passage_lists_texts = np.array(passage_lists_texts).reshape(len(queries), list_length)
    passage_lists_ids = np.array(passage_lists_ids).reshape(len(queries), list_length)
    passage_lists_scores = np.array(passage_lists_scores).reshape(len(queries), list_length)

    return queries, passage_lists_texts, passage_lists_ids, passage_lists_scores

class Trainer:
    def __init__(self, model, optimizer, scheduler, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.best_overall_dev_loss = float('inf')
        self.best_rank_dev_loss = float('inf')
        self.best_contrastive_dev_loss = float('inf')
        self.non_improvement_count = 0
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

    @cat_input_tensor
    def rank_loss(self, query_embeddings, all_passage_embeddings, all_passage_scores):
        bs = len(query_embeddings)
        all_passage_embeddings = all_passage_embeddings.reshape(bs, self.config.list_length, -1)
        y_preds = torch.bmm(all_passage_embeddings, query_embeddings.unsqueeze(-1)).squeeze(-1).double()
        all_passage_scores = torch.tensor(all_passage_scores).cuda().reshape(bs, self.config.list_length)
        rank_temp = torch.tensor(self.config.rank_temp).double()
        kl_temp = torch.tensor(self.config.kl_temp).double()

        y_preds = F.log_softmax(y_preds / rank_temp, dim=-1)        
        all_passage_scores = F.log_softmax(all_passage_scores / kl_temp, dim=-1)

        loss = self.kl_loss(y_preds, all_passage_scores).float()
        return loss

    @cat_input_tensor
    def contrastive_loss(self, query_embeddings, all_passage_embeddings, all_ids, all_passage_scores):
        num_queries = len(query_embeddings)
        assert len(all_passage_embeddings) == num_queries * self.config.list_length
        assert len(all_passage_embeddings) == len(all_passage_scores)
        pos_passage_embeddings = []
        hard_negative_passage_embeddings = []
        pos_passage_index_map = {}
        pos_passage_curr_index = 0
        hn_passage_index_map = {}
        hn_passage_curr_index = 0

        no_contrast_ids = []
        curr_no_contrast_ids = []
        top_passage_score = 0

        for i in range(len(all_passage_embeddings)):
            if i % self.config.list_length == 0:
                top_passage_score = all_passage_scores[i]
                pos_passage_embeddings.append(all_passage_embeddings[i])
                if all_ids[i] in pos_passage_index_map:
                    pos_passage_index_map[all_ids[i]].append(pos_passage_curr_index)
                else:
                    pos_passage_index_map[all_ids[i]] = [pos_passage_curr_index]
                pos_passage_curr_index += 1
                if i != 0:
                    no_contrast_ids.append(curr_no_contrast_ids)
                curr_no_contrast_ids = []
                curr_no_contrast_ids.append(all_ids[i])
            else:
                current_score = all_passage_scores[i]
                hard_negative_passage_embeddings.append(all_passage_embeddings[i])
                if all_ids[i] in hn_passage_index_map:
                    hn_passage_index_map[all_ids[i]].append(hn_passage_curr_index)
                else:
                    hn_passage_index_map[all_ids[i]] = [hn_passage_curr_index]
                hn_passage_curr_index += 1
                if current_score > self.config.threshold_score * top_passage_score:
                    curr_no_contrast_ids.append(all_ids[i])

        if len(curr_no_contrast_ids) != 0:
            no_contrast_ids.append(curr_no_contrast_ids)
        
        pos_passage_embeddings = torch.stack(pos_passage_embeddings)
        hard_negative_passage_embeddings = torch.stack(hard_negative_passage_embeddings)

        assert len(no_contrast_ids) == num_queries
        assert len(pos_passage_embeddings) == num_queries
        assert hn_passage_curr_index == len(hard_negative_passage_embeddings)
        assert pos_passage_curr_index == len(pos_passage_embeddings)

        pos_not_contrast_mask = torch.ones(len(pos_passage_embeddings), num_queries).cuda()
        for i in range(len(no_contrast_ids)):
            for doc_id in no_contrast_ids[i]:
                if doc_id in pos_passage_index_map:
                    for non_contrast_index in pos_passage_index_map[doc_id]:
                        if non_contrast_index != i:
                            pos_not_contrast_mask[non_contrast_index, i] = 0

        hn_not_contrast_mask = torch.ones(len(hard_negative_passage_embeddings), num_queries).cuda()
        for i in range(len(no_contrast_ids)):
            for doc_id in no_contrast_ids[i]:
                if doc_id in hn_passage_index_map:
                    for non_contrast_index in hn_passage_index_map[doc_id]:
                        hn_not_contrast_mask[non_contrast_index, i] = 0

        temp = torch.tensor(self.config.temp).double()
        pos_query_scores = torch.matmul(pos_passage_embeddings, query_embeddings.T).double()
        pos_query_scores = torch.exp(pos_query_scores / temp)
        pos_query_scores = pos_query_scores * pos_not_contrast_mask

        hn_query_scores = torch.matmul(hard_negative_passage_embeddings, query_embeddings.T).double()
        hn_query_scores = torch.exp(hn_query_scores / temp)
        hn_query_scores = hn_query_scores * hn_not_contrast_mask

        losses = torch.diagonal(pos_query_scores) / (torch.sum(pos_query_scores, dim=0) + torch.sum(hn_query_scores, dim=0))

        losses = torch.log(losses).float()
        return -losses.mean()

    def train(self, train_dataloader):
        self.model.train()
        cache_query_embeddings = []
        cache_all_passage_embeddings = []
        cache_all_docids = []
        cache_all_scores = []
        closures_query = []
        closures_all_passage = []

        running_contrastive_train_loss = 0.0
        running_rank_train_loss = 0.0
        num_train_batches = 0

        self.optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_dataloader)):
            (
                query_inputids,
                query_mask,
                all_passage_inputids,
                all_passage_masks,
                all_passage_docids,
                all_passage_scores,
            ) = batch

            query_embeddings, cq = call_model(self.model, query_inputids.cuda(), query_mask.cuda())
            all_passage_embeddings, cap = call_model(self.model, all_passage_inputids.cuda(), all_passage_masks.cuda())

            cache_query_embeddings.append(query_embeddings)
            cache_all_passage_embeddings.append(all_passage_embeddings)
            cache_all_docids.extend(all_passage_docids)
            cache_all_scores.extend(all_passage_scores)

            closures_query.append(cq)
            closures_all_passage.append(cap)

            if (step + 1) % self.config.accumulation_steps == 0:
                num_train_batches += 1

                contrastive_loss_val = self.contrastive_loss(
                    cache_query_embeddings,
                    cache_all_passage_embeddings,
                    cache_all_docids,
                    cache_all_scores
                )
                listwise_loss_val = self.rank_loss(
                    cache_query_embeddings,
                    cache_all_passage_embeddings,
                    cache_all_scores,
                )

                loss = contrastive_loss_val * self.config.contrastive_loss_weight + listwise_loss_val
                loss.backward()

                for f, r in zip(closures_query, cache_query_embeddings):
                    f(r)
                for f, r in zip(closures_all_passage, cache_all_passage_embeddings):
                    f(r)

                cache_query_embeddings = []
                cache_all_passage_embeddings = []
                cache_all_docids = []
                cache_all_scores = []
                closures_query = []
                closures_all_passage = []

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                running_contrastive_train_loss += contrastive_loss_val.detach().item()
                running_rank_train_loss += listwise_loss_val.detach().item()

        avg_contrastive_loss = running_contrastive_train_loss / num_train_batches
        avg_rank_loss = running_rank_train_loss / num_train_batches

        print("TRAIN CONTRASTIVE LOSS", avg_contrastive_loss)
        print("TRAIN RANK LOSS", avg_rank_loss)

    def evaluate(self, dev_dataloader):
        self.model.eval()
        cache_query_embeddings = []
        cache_all_passage_embeddings = []
        cache_all_docids = []
        cache_all_scores = []
        total_contrastive_loss = 0.0
        total_listwise_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for step, batch in enumerate(tqdm(dev_dataloader)):
                (
                    query_inputids,
                    query_mask,
                    all_passage_inputids,
                    all_passage_masks,
                    all_passage_docids,
                    all_passage_scores,
                ) = batch

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    query_embeddings = self.model(query_inputids.cuda(), query_mask.cuda())
                    all_passage_embeddings = self.model(all_passage_inputids.cuda(), all_passage_masks.cuda())

                cache_query_embeddings.append(query_embeddings)
                cache_all_passage_embeddings.append(all_passage_embeddings)
                cache_all_docids.extend(all_passage_docids)
                cache_all_scores.extend(all_passage_scores)

                if (step + 1) % self.config.accumulation_steps == 0 or ((step < self.config.accumulation_steps) and ((step + 1) == len(dev_dataloader))):
                    num_batches += 1

                    contrastive_loss_val = self.contrastive_loss(
                        cache_query_embeddings,
                        cache_all_passage_embeddings,
                        cache_all_docids,
                        cache_all_scores
                    ).detach().cpu()

                    listwise_loss_val = self.rank_loss(
                        cache_query_embeddings,
                        cache_all_passage_embeddings,
                        cache_all_scores,
                    ).detach().cpu()

                    total_contrastive_loss += contrastive_loss_val.item()
                    total_listwise_loss += listwise_loss_val.item()

                    cache_query_embeddings = []
                    cache_all_passage_embeddings = []
                    cache_all_docids = []
                    cache_all_scores = []

        avg_contrastive_loss = total_contrastive_loss / num_batches
        avg_rank_loss = total_listwise_loss / num_batches

        print("CONTRASTIVE DEV LOSS", avg_contrastive_loss)
        print("LISTWISE DEV LOSS", avg_rank_loss)

        return avg_contrastive_loss, avg_rank_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='msmarco', help='Dataset name')
    args = parser.parse_args()

    config = Config(dataset=args.dataset)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

    emb_model = EmbeddingModel(config.model_name_or_path, dropout=config.dropout).cuda()
    optimizer = torch.optim.AdamW(emb_model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    print("BEGIN LOADING TRAINING DATA")
   
    train_queries, train_passage_texts, train_passage_ids, train_passage_scores = load_dataset(
        config.train_file_path, config.list_length
    )
    dev_queries, dev_passage_texts, dev_passage_ids, dev_passage_scores = load_dataset(
        config.dev_file_path, config.list_length
    )

    train_dataset = RankingDataset(
        train_queries, train_passage_texts, train_passage_ids, train_passage_scores, config.list_length
    )
    dev_dataset = RankingDataset(
        dev_queries, dev_passage_texts, dev_passage_ids, dev_passage_scores, config.list_length
    )

    collator_function = Collator(
        tokenizer, config.instruction, config.query_maxlength, config.text_maxlength, config.list_length
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=collator_function,
        num_workers=4,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=config.batch_size,
        collate_fn=collator_function,
        num_workers=4,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )

    def create_scheduler(optimizer, train_dataloader, batch_size, base_lr, total_epochs=20):
        total_steps_in_epoch = len(train_dataloader) // config.accumulation_steps
        warmup_steps = max(10, total_steps_in_epoch // 2)
        total_steps = total_epochs * total_steps_in_epoch
        decay_steps = total_steps - warmup_steps

        def warmup_then_decay(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(warmup_steps)
            else:
                decay_factor = (current_step - warmup_steps) / decay_steps
                return max(0.1, 1.0 - decay_factor)

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_then_decay)
        return lr_scheduler

    lr_scheduler = create_scheduler(
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        batch_size=config.batch_size,
        base_lr=config.lr,
        total_epochs=config.num_epochs,
    )

    trainer = Trainer(emb_model, optimizer, lr_scheduler, config)

    for epoch in range(config.num_epochs + 1):
        print(f"Epoch {epoch}/{config.num_epochs}")
        avg_contrastive_loss, avg_rank_loss = trainer.evaluate(dev_dataloader)

        trainer.non_improvement_count += 1

        if (avg_rank_loss < trainer.best_rank_dev_loss):
            trainer.best_rank_dev_loss = avg_rank_loss
            trainer.non_improvement_count = 0

        if ((avg_contrastive_loss*config.contrastive_loss_weight + avg_rank_loss) < trainer.best_overall_dev_loss):
            trainer.best_overall_dev_loss = avg_contrastive_loss*config.contrastive_loss_weight + avg_rank_loss
            trainer.non_improvement_count = 0

            if config.save_model:
                model_save_name = (
                    f"{config.save_model_name}"
                )
                emb_model.bert.save_pretrained(model_save_name)
                print(f"Saved model to {model_save_name}")
                
        if trainer.non_improvement_count >= 2:
            print("Early stopping due to no improvement in validation loss.")
            break

        if epoch == config.num_epochs:
            break

        trainer.train(train_dataloader)

if __name__ == "__main__":
    main()
