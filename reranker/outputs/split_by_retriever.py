import json
from tqdm import tqdm
from datasets import load_dataset

beir_datasets = ['msmarco', 'fiqa', 'scifact', 'trec-covid', 'nfcorpus', 'arguana', 'webis-touche2020', 'dbpedia-entity', 'scidocs', 'climate-fever']
query_types = ['titles', 'claims', 'questions', 'random', 'msmarco', 'keywords']

retrievers = {"bge": "BAAI_bge-base-en-v1.5", "gte": "Alibaba-NLP_gte-base-en-v1.5", "snowflake": "Snowflake_snowflake-arctic-embed-m-v1.5"}

for beir_dataset in tqdm(beir_datasets):
    for query_type in tqdm(query_types):
        for retriever in retrievers.keys():
            
            retriever_full_name = retrievers[retriever]

            ranked_output_file = f'rankt5-{beir_dataset}-queries-{query_type}.jsonl'
            retrieval_file = f'../../query_generation/retrieval_runs/run.{retriever_full_name}.{beir_dataset}.generated-queries-{query_type}.filtered_20.txt'

            retrieval_dict = {}
            with open(retrieval_file, 'r') as input_retrieval_file:
                for line in input_retrieval_file:
                    qid, _, pid, rank, score, _ = line.split()
                    if qid in retrieval_dict:
                        retrieval_dict[qid].append(pid)
                    else:
                        retrieval_dict[qid] = [pid]
            
            new_ranked_output_file = retriever + '_' + ranked_output_file
            with open(ranked_output_file, 'r', encoding='utf-8') as ranked_input:
                with open(new_ranked_output_file, 'w', encoding='utf-8') as ranked_output:
                    for line in ranked_input:
                        ranked_dict = json.loads(line)
                        qid = ranked_dict['qid']
                        if qid not in retrieval_dict:
                            continue
                        new_passages_list = []
                        for passage in ranked_dict['passages']:
                            if passage['docid'] in retrieval_dict[qid]:
                                new_passages_list.append(passage)
                        ranked_dict['passages'] = new_passages_list
                        ranked_output.write(json.dumps(ranked_dict, ensure_ascii=False) + '\n')
