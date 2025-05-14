import json
import random
from tqdm import tqdm
import os

def load_samples(input_files):
    samples = []
    total_count = 0

    for file_name in input_files:
        if not os.path.exists(file_name):
            print(f"Warning: {file_name} does not exist. Skipping.")
            continue

        with open(file_name, 'r', encoding='utf-8') as input_f:
            for line in tqdm(input_f, desc=f'Loading {file_name}'):
                total_count += 1
                try:
                    jsonl_line = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {file_name}: {e}")
                    continue

                samples.append(jsonl_line)

    print(f"Total samples loaded: {total_count}")
    print(f"Total valid samples: {len(samples)}")

    return samples

def split_train_dev(samples, train_file, dev_file, train_ratio=0.9, seed=42):
    """
    Splits samples into training and development sets based on unique 'qid's.

    Args:
        samples (list): List of JSON objects.
        train_file (str): Output filename for training samples.
        dev_file (str): Output filename for development samples.
        train_ratio (float): Proportion of data to include in the training set.
        seed (int): Random seed for reproducibility.
    """
    # Shuffle the samples to ensure randomness
    random.seed(seed)
    random.shuffle(samples)

    # Extract unique qids
    unique_qids = list({sample['qid'] for sample in samples})
    random.shuffle(unique_qids)

    num_train_qids = int(train_ratio * len(unique_qids))
    train_qids = set(unique_qids[:num_train_qids])
    dev_qids = set(unique_qids[num_train_qids:])

    print(f"Number of unique qids: {len(unique_qids)}")
    print(f"Training qids: {len(train_qids)}")
    print(f"Development qids: {len(dev_qids)}")

    # Write to train and dev files
    with open(train_file, 'w', encoding='utf-8') as train_f, \
         open(dev_file, 'w', encoding='utf-8') as dev_f:
        
        for sample in tqdm(samples, desc='Writing train/dev samples'):
            qid = sample.get('qid')
            if qid in train_qids:
                train_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            elif qid in dev_qids:
                dev_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            else:
                print(f"Warning: qid {qid} not found in train or dev sets.")

    print(f"Training and development files have been created:\n- {train_file}\n- {dev_file}")

def main():
    beir_datasets = ['msmarco', 'fiqa', 'scifact', 'trec-covid', 'nfcorpus', 'arguana', 'webis-touche2020', 'dbpedia-entity', 'scidocs', 'climate-fever']
    retrievers = {"bge": "BAAI_bge-base-en-v1.5", "gte": "Alibaba-NLP_gte-base-en-v1.5", "snowflake": "Snowflake_snowflake-arctic-embed-m-v1.5"}

    for beir_dataset in beir_datasets:  
        for retriever in retrievers.keys():      
            # Define input filenames
            input_files = [
                f'outputs/{retriever}_{beir_dataset}-queries-questions-normalized.jsonl',
                f'outputs/{retriever}_{beir_dataset}-queries-claims-normalized.jsonl',
                f'outputs/{retriever}_{beir_dataset}-queries-titles-normalized.jsonl',
                f'outputs/{retriever}_{beir_dataset}-queries-msmarco-normalized.jsonl',
                f'outputs/{retriever}_{beir_dataset}-queries-random-normalized.jsonl',
                f'outputs/{retriever}_{beir_dataset}-queries-keywords-normalized.jsonl'
            ]
            
            # Define output filenames
            train_output_file = f'beir.{retriever}.{beir_dataset}.train.generated_queries.listwise.jsonl'
            dev_output_file = f'beir.{retriever}.{beir_dataset}.dev.generated_queries.listwise.jsonl'

            # Step 1: Load samples
            samples = load_samples(input_files)
            
            # Step 2: Split into train and dev sets
            split_train_dev(samples, train_output_file, dev_output_file)

if __name__ == "__main__":
    main()
