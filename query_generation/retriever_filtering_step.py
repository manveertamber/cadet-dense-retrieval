from tqdm import tqdm

filter_k = 20
beir_datasets = ['msmarco', 'fiqa', 'scifact', 'trec-covid', 'nfcorpus', 'arguana', 'webis-touche2020', 'dbpedia-entity', 'scidocs', 'climate-fever']
query_types = ['keywords', 'titles', 'claims', 'questions', 'random', 'msmarco']
retrievers = ['Snowflake_snowflake-arctic-embed-m-v1.5', 'BAAI_bge-base-en-v1.5', 'Alibaba-NLP_gte-base-en-v1.5']

for retriever in retrievers:
    for beir_dataset in beir_datasets:
        for query_type in query_types:
            disregard_ids = set()
            with open('duplicate_ids/' + beir_dataset + '_duplicate_ids.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    disregard_ids.add(line.strip())
            with open('test_ids/' + beir_dataset + '_test_ids.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    disregard_ids.add(line.strip())

            current_query = -1
            rank = 1
            step_one_filtered_lines = []
            with open(f'retrieval_runs/run.{retriever}.{beir_dataset}.generated-queries-{query_type}_100.txt', 'r') as f:
                for line in tqdm(f):
                    vals = line.split(' ')
                    qid = str(vals[0])
                    pid = str(vals[2])

                    if qid in disregard_ids:
                        continue

                    if current_query != qid:                
                        current_query = qid
                        rank = 1

                    if (pid not in disregard_ids):
                        vals[3] = str(rank)
                        rank += 1
                        step_one_filtered_lines.append(' '.join(vals).strip())

            current_query = -1
            rank = 1
            keep_query = False
            buffer = []
            with open(f'retrieval_runs/run.{retriever}.{beir_dataset}.generated-queries-{query_type}.filtered.txt', 'w') as filtered_f:
                for line in tqdm(step_one_filtered_lines):
                    vals = line.split(' ')
                    qid = str(vals[0])
                    pid = str(vals[2])
                    rank = int(vals[3])

                    if current_query != qid:
                        if keep_query == True:
                            for buffer_line in buffer:
                                filtered_f.write(buffer_line + '\n')

                        buffer = []
                        current_query = qid
                        keep_query = False

                    if (qid == pid) and rank <= filter_k:
                        keep_query = True
                    
                    buffer.append(line.strip())

                if keep_query == True:
                    for buffer_line in buffer:
                        filtered_f.write(buffer_line + '\n')