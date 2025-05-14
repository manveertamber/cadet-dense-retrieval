# sample up to 100k passages for each corpora, after getting rid of filter_ids, test_ids
# in the case of msmarco, only sample passages that are in the training set (meaning these passages are relevant to a query in the training set)

from datasets import load_dataset
from tqdm import tqdm
import random
import re

# Function to load ids from a file
def load_ids(file_path):
    filter_ids = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            filter_ids.add(line.strip())
    return filter_ids

# Function to process the dataset, filter, and sample passages
def process_and_sample(ds, id_key, filter_ids, n_samples=100000, train_ids=None):
    passages_ids = []
    filtered_count = 0

    for line in ds:
        _id = str(line[id_key])
        text = str(line['text']).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()[:2048]
        
        if _id not in filter_ids:
            passages_ids.append(_id)
        else:
            filtered_count += 1

    if train_ids is not None:
        old_passage_ids = passages_ids
        passages_ids = []
        for id in old_passage_ids:
            if id in train_ids:
                passages_ids.append(id)

    # Limit to the required number of ids
    n = min(n_samples, len(passages_ids))
    chosen_ids = random.sample(passages_ids, n)
    
    return chosen_ids, filtered_count

# Function to write chosen ids to file
def write_chosen_ids(chosen_ids, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for id in chosen_ids:
            f.write(id + '\n')

# Process BEIR datasets
beir_datasets = ['msmarco', 'trec-covid', 'nfcorpus', 'fiqa', 'arguana', 'webis-touche2020', 
                 'dbpedia-entity', 'scidocs', 'climate-fever', 'scifact']

for beir_dataset in beir_datasets:
    print(f"Processing BEIR dataset: {beir_dataset}")
    print("__________________________")
    
    # Load dataset and filter_ids
    ds = load_dataset(f"BeIR/{beir_dataset}", "corpus")['corpus']
    filter_ids = set()
    duplicate_ids = load_ids(f'duplicate_ids/{beir_dataset}_duplicate_ids.txt')
    test_ids = load_ids(f'test_ids/{beir_dataset}_test_ids.txt')
    filter_ids = duplicate_ids.union(test_ids)
    train_ids = None

    n_samples = 100000

    if beir_dataset == 'msmarco':
        train_ids = load_ids('train_ids/msmarco_train_ids.txt')
        n_samples = 200000

    # Process and sample passages
    chosen_ids, filtered_count = process_and_sample(ds, '_id', filter_ids, n_samples=n_samples, train_ids=train_ids)
    
    # Write chosen sample IDs to file
    write_chosen_ids(chosen_ids, f'chosen_ids/{beir_dataset}_chosen_ids.txt')

