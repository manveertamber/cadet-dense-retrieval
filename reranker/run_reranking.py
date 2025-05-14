import argparse
import jsonlines
import torch
from tqdm import tqdm
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    T5ForConditionalGeneration,
    AutoModelForCausalLM
)

def read_jsonl(path):
    data = []
    with jsonlines.open(path, mode='r') as reader:
        for line in reader:
            data.append(line)
    return data

def write_jsonl(path, data):
    with jsonlines.open(path, mode='w') as writer:
        writer.write_all(data)
    print(f"Written output to {path}")

class RerankerModel(torch.nn.Module):
    def __init__(self, model_name, device='cuda'):
        super().__init__()
        self.device = device
        self.model_name = model_name.lower()

        # Some known token IDs used by monot5/rankt5
        self.monot5_true_false_tokens = [1176, 6136]  # "▁true", "▁false"
        self.rankt5_extra_id_10 = 32089              # <extra_id_10>

        if 'rankt5' in self.model_name:
            self.model_type = 'rankt5'
            self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        elif 'monot5' in self.model_name:
            self.model_type = 'monot5'
            self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        elif 'bge-reranker' in self.model_name:
            self.model_type = 'bge_reranker'
            # BGE-specific
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.tokenizer.padding_side = 'right'
            self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16).to(self.device).eval()
        else:
            # Fallback to sequence classifier (e.g., cross-encoder)
            self.model_type = 'sequence_classifier'
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, batch):
        if self.model_type == 'monot5':
            # monot5
            output = self.model.generate(**batch, max_length=2, return_dict_in_generate=True, output_scores=True)
            scores_tensor = torch.stack(output.scores)
            log_probs = torch.nn.functional.log_softmax(scores_tensor[0][:, self.monot5_true_false_tokens], dim=1)
            scores = log_probs[:, 0].tolist()

        elif self.model_type == 'rankt5':
            # rankt5
            output = self.model.generate(**batch, max_length=2, return_dict_in_generate=True, output_scores=True)
            scores_tensor = torch.stack(output.scores)
            scores = scores_tensor[0][:, self.rankt5_extra_id_10].tolist()

        elif self.model_type == 'bge_reranker':
            # bge-reranker
            query_lengths = batch.pop("query_lengths")
            prompt_lengths = batch.pop("prompt_lengths")

            outputs = self.model(
                **batch,
                return_dict=True,
                cutoff_layers=[25], 
                compress_ratio=2,
                compress_layer=[8],
                query_lengths=query_lengths,
                prompt_lengths=prompt_lengths
            )

            scores = []
            for i in range(len(outputs.logits)):
                if hasattr(outputs, "attention_masks"):
                    mask = outputs.attention_masks[i]
                else:
                    mask = batch["attention_mask"][i] if "attention_mask" in batch else None

                logits = self.last_logit_pool(outputs.logits[i], mask)
                scores.extend(logits.cpu().float().tolist())

        else:
            logits = self.model(**batch).logits
            if logits.shape[1] == 1:
                scores = logits.squeeze(dim=1).tolist()
            else:
                scores = logits[:, 0].tolist()

        return scores

    def score_pairs(self, pairs, batch_size, max_length):
        """
        pairs: list of (query, passage)
        """
        scores = []

        if self.model_type == 'monot5':
            # monot5 style input
            input_texts = [f"Query: {q} Document: {d} Relevant:" for (q, d) in pairs]
            encodings = self.tokenizer(
                input_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)

            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(self.device == 'cuda')):
                    for start in tqdm(range(0, len(pairs), batch_size), desc="Scoring"):
                        end = start + batch_size
                        batch_encodings = {k: v[start:end] for k, v in encodings.items()}
                        batch_scores = self.forward(batch_encodings)
                        scores.extend(batch_scores)

        elif self.model_type == 'rankt5':
            # rankt5 style input
            input_texts = [f"Query: {q} Document: {d}" for (q, d) in pairs]
            encodings = self.tokenizer(
                input_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)

            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(self.device == 'cuda')):
                    for start in tqdm(range(0, len(pairs), batch_size), desc="Scoring"):
                        end = start + batch_size
                        batch_encodings = {k: v[start:end] for k, v in encodings.items()}
                        batch_scores = self.forward(batch_encodings)
                        scores.extend(batch_scores)

        elif self.model_type == 'bge_reranker':
            # bge-reranker
            with torch.no_grad():
                for start in tqdm(range(0, len(pairs), batch_size), desc="Scoring"):
                    end = start + batch_size
                    batch_pairs = pairs[start:end]
                    # Build special BGE-style inputs
                    inputs, query_lengths, prompt_lengths = self.get_inputs(
                        batch_pairs, self.tokenizer, max_length=max_length
                    )
                    inputs = inputs.to(self.device)
                    # Add query/prompt lengths for the forward pass
                    inputs["query_lengths"] = query_lengths
                    inputs["prompt_lengths"] = prompt_lengths

                    batch_scores = self.forward(inputs)
                    scores.extend(batch_scores)

        else:
            # Standard sequence classifier (e.g. cross-encoder)
            q_texts = [p[0] for p in pairs]
            d_texts = [p[1] for p in pairs]
            encodings = self.tokenizer(
                q_texts,
                d_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)

            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(self.device == 'cuda')):
                    for start in tqdm(range(0, len(pairs), batch_size), desc="Scoring"):
                        end = start + batch_size
                        batch_encodings = {k: v[start:end] for k, v in encodings.items()}
                        batch_scores = self.forward(batch_encodings)
                        scores.extend(batch_scores)

        return scores

    @staticmethod
    def last_logit_pool(logits: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Pools the final logit from the last non-padded token.
        If no attention_mask is provided, it defaults to the last token in `logits`.
        """
        if attention_mask is None:
            # If there's no attention_mask, pick the last token logit
            return logits[:, -1]

        # If attention_mask is shape [seq_len], expand to match logits shape
        if len(attention_mask.shape) == 1:
            attention_mask = attention_mask.unsqueeze(0)

        # sequence_lengths = sum of mask - 1 (last token index)
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = logits.shape[0]
        return torch.stack([logits[i, sequence_lengths[i]] for i in range(batch_size)], dim=0)

    @staticmethod
    def get_inputs(pairs, tokenizer, prompt=None, max_length=1024):
        """
        Prepares the specialized input format for the bge-reranker.
        By default, uses a short prompt. Adjust as desired.
        """
        if prompt is None:
            prompt = "Predict whether passage B contains an answer to query A."
        sep = "\n"

        prompt_ids = tokenizer(prompt, return_tensors=None, add_special_tokens=False)['input_ids']
        sep_ids = tokenizer(sep, return_tensors=None, add_special_tokens=False)['input_ids']
        bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []

        inputs_list = []
        query_lengths = []
        prompt_lengths = []
        for query, passage in pairs:
            # Create "A: <query>" tokens
            query_inputs = tokenizer(
                f"A: {query}",
                return_tensors=None,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True
            )
            # Create "B: <passage>" tokens
            passage_inputs = tokenizer(
                f"B: {passage}",
                return_tensors=None,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True
            )

            # Combine them into a single set of input_ids
            # first half: [BOS] + query_tokens
            # second half: sep + passage_tokens
            item = tokenizer.prepare_for_model(
                bos + query_inputs['input_ids'],
                sep_ids + passage_inputs['input_ids'],
                truncation='only_second',
                max_length=max_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False
            )
            # Then add the prompt at the end: 
            #  item['input_ids'] + sep + prompt
            item['input_ids'] = item['input_ids'] + sep_ids + prompt_ids
            item['attention_mask'] = [1] * len(item['input_ids'])

            inputs_list.append(item)
            query_lengths.append(len(bos + query_inputs['input_ids'] + sep_ids))
            prompt_lengths.append(len(sep_ids + prompt_ids))

        # Now pad all items to the same length
        padded_inputs = tokenizer.pad(
            inputs_list,
            padding=True,
            max_length=max_length + len(sep_ids) + len(prompt_ids),
            pad_to_multiple_of=8,
            return_tensors='pt'
        )

        return padded_inputs, query_lengths, prompt_lengths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True,
                        help=("Reranker model name or path "
                              "(e.g. 'castorini/monot5-base-msmarco-10k', "
                              "'Soyoung97/RankT5-base', "
                              "'bge-reranker-v2.5-gemma2-lightweight', "
                              "or 'cross-encoder/ms-marco-MiniLM-L-12-v2')."))
    parser.add_argument('--input_path', required=True, help="Path to input JSONL.")
    parser.add_argument('--output_path', required=True, help="Path to output JSONL.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for scoring.")
    parser.add_argument('--topk', type=int, default=100, help="Number of top passages to re-rank.")
    parser.add_argument('--max_input_length', type=int, default=512, help="Maximum input length for tokenization.")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use (e.g., 'cuda', 'cpu').")
    args = parser.parse_args()

    query_key = "query"
    passages_key = "passages"
    text_key = "text"
    score_key = "score"

    data = read_jsonl(args.input_path)

    reranker = RerankerModel(model_name=args.model_name, device=args.device)

    output_data = []
    for entry in tqdm(data, desc="Re-ranking"):
        query = entry[query_key]
        passages = entry[passages_key]
        passages = passages[:args.topk]

        # Build the (query, passage_text) list
        pairs = [(query, p[text_key]) for p in passages]

        # Get the scores
        scores = reranker.score_pairs(
            pairs,
            batch_size=args.batch_size,
            max_length=args.max_input_length
        )

        # Attach the new scores and re-sort
        for p, s in zip(passages, scores):
            p[score_key] = s
        passages = sorted(passages, key=lambda x: x[score_key], reverse=True)

        entry[passages_key] = passages
        output_data.append(entry)

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_path, output_data)
    print("Done!")

if __name__ == "__main__":
    main()
