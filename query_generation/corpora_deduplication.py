from datasets import load_dataset
from tqdm import tqdm
import re
import unicodedata

def preprocess_text(text):
    # Normalize Unicode characters to NFKC
    text = unicodedata.normalize('NFKC', text)
    # Remove accents and diacritics
    text = ''.join(
        char for char in unicodedata.normalize('NFD', text)
        if not unicodedata.combining(char)
    )
    # Convert to lowercase
    text = text.lower()
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation while preserving whitespace
    text = re.sub(r'[^\w\s]', '', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text

# Function to deduplicate based on exact or substring matches
def deduplicate_passages(passages_tuples):
    filtered_passage_ids = set()

    for i in (range(len(passages_tuples))):
        current_id, current_text = passages_tuples[i]
        if len(current_text) < 10:
            filtered_passage_ids.add(current_id)

    sorted_passages_tuples = sorted(passages_tuples, key=lambda x: x[1])

    for i in (range(len(sorted_passages_tuples) - 1)):
        current_id, current_text = sorted_passages_tuples[i]
        for j in range(i+1, min(len(sorted_passages_tuples), i+10)):
            next_id, next_text = sorted_passages_tuples[j]
            if current_text in next_text:
                filtered_passage_ids.add(current_id)

    sorted_passages_tuples = sorted(passages_tuples, key=lambda x: x[1][::-1])

    for i in (range(len(sorted_passages_tuples) - 1)):
        current_id, current_text = sorted_passages_tuples[i]
        for j in range(i+1, min(len(sorted_passages_tuples), i+10)):
            next_id, next_text = sorted_passages_tuples[j]
            if current_text in next_text:
                filtered_passage_ids.add(current_id)

    return filtered_passage_ids

# BEIR
beir_datasets = ['trec-covid', 'nfcorpus', 'fiqa', 'arguana', 'webis-touche2020', 'dbpedia-entity', 'scidocs', 'climate-fever', 'scifact', 'msmarco']
for beir_dataset in beir_datasets:
    print("beir_dataset", beir_dataset)
    # Load the dataset
    ds = load_dataset("BeIR/" + beir_dataset, "corpus")

    passages_tuples = []

    # Process the passages
    for line in tqdm(ds['corpus']):
        _id = (str(line['_id']))
        text = str(line['text']).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()[:2048]
        text = preprocess_text(text)
        passages_tuples.append((_id, text))

    # Perform deduplication
    duplicate_ids = deduplicate_passages(passages_tuples)

    print(f"Filtered {len(duplicate_ids)} duplicate passages from {len(passages_tuples)} {beir_dataset} passages")
    print("__________________________")

    with open('duplicate_ids/' + beir_dataset + '_duplicate_ids.txt', 'w', encoding='utf-8') as f:
        for x in duplicate_ids:
            f.write(x + '\n')
