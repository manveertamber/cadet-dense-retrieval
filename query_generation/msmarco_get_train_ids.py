train_passage_ids = set()
with open('msmarco.qrels.train.tsv', 'r', encoding='utf-8') as f:
    for line in f:
        vals = line.strip().split('\t')
        train_passage_ids.add(vals[2].strip())

with open('train_ids/msmarco_train_ids.txt', 'w', encoding='utf-8') as output_f:
    for id in train_passage_ids:
        output_f.write(id + '\n')
