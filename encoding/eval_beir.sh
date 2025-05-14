for dataset in 'trec-covid' 'nfcorpus' 'fiqa' 'scidocs' 'arguana' 'webis-touche2020'  'dpbedia-entity' 'climate-fever' 'scifact'; do
    model_name=${dataset}_bge_embedding_model

    python3 encode_corpus.py --model_name models/${model_name} --normalize --pooling cls --batch_size 1800 --dataset ${dataset}

    python -m pyserini.search.faiss \
    --threads 16 --batch-size 512 \
    --encoder-class auto \
    --encoder models/${model_name} --l2-norm --query-prefix "Represent this sentence for searching relevant passages: " \
    --index indices/models_${model_name}_${dataset}_index \
    --topics beir-v1.0.0-${dataset}-test \
    --output run.beir.${model_name}.${dataset}.txt \
    --hits 1000 --remove-query

    python -m pyserini.eval.trec_eval \
    -c -m ndcg_cut.10 beir-v1.0.0-${dataset}-test \
    run.beir.${model_name}.${dataset}.txt

    python -m pyserini.eval.trec_eval \
    -c -m recall.100 beir-v1.0.0-${dataset}-test \
    run.beir.${model_name}.${dataset}.txt

    rm -r indices/models_${model_name}_${dataset}_index

    model_name=${dataset}_snowflake_embedding_model

    python3 encode_corpus.py --model_name models/${model_name} --normalize --pooling cls --batch_size 1800 --dataset ${dataset}

    python -m pyserini.search.faiss \
    --threads 16 --batch-size 512 \
    --encoder-class auto \
    --encoder models/${model_name} --l2-norm --query-prefix "Represent this sentence for searching relevant passages: " \
    --index indices/models_${model_name}_${dataset}_index \
    --topics beir-v1.0.0-${dataset}-test \
    --output run.beir.${model_name}.${dataset}.txt \
    --hits 1000 --remove-query


    python -m pyserini.eval.trec_eval \
    -c -m ndcg_cut.10 beir-v1.0.0-${dataset}-test \
    run.beir.${model_name}.${dataset}.txt

    python -m pyserini.eval.trec_eval \
    -c -m recall.100 beir-v1.0.0-${dataset}-test \
    run.beir.${model_name}.${dataset}.txt

    rm -r indices/models_${model_name}_${dataset}_index

    model_name=${dataset}_gte_embedding_model

    python3 encode_corpus.py --model_name models/${model_name} --normalize --pooling cls --batch_size 1800 --dataset ${dataset}
    
    python -m pyserini.search.faiss \
    --threads 16 --batch-size 512 \
    --encoder-class auto \
    --encoder models/${model_name} --l2-norm --query-prefix "" \
    --index indices/models_${model_name}_${dataset}_index \
    --topics beir-v1.0.0-${dataset}-test \
    --output run.beir.${model_name}.${dataset}.txt \
    --hits 1000 --remove-query


    python -m pyserini.eval.trec_eval \
    -c -m ndcg_cut.10 beir-v1.0.0-${dataset}-test \
    run.beir.${model_name}.${dataset}.txt

    python -m pyserini.eval.trec_eval \
    -c -m recall.100 beir-v1.0.0-${dataset}-test \
    run.beir.${model_name}.${dataset}.txt

    rm -r indices/models_${model_name}_${dataset}_index

    model='e5'
    python3 encode_corpus_e5.py --model_name models/${model_name} --normalize --pooling mean --batch_size 1800 --dataset ${dataset}
    python -m pyserini.search.faiss \
    --threads 16 --batch-size 512 \
    --encoder-class auto \
    --encoder models/${model_name} --pooling mean --l2-norm --query-prefix "query: " \
    --index indices/models_${model_name}_${dataset}_index \
    --topics beir-v1.0.0-${dataset}-test \
    --output run.beir.${model_name}.${dataset}.txt \
    --hits 1000 --remove-query


    python -m pyserini.eval.trec_eval \
    -c -m ndcg_cut.10 beir-v1.0.0-${dataset}-test \
    run.beir.${model_name}.${dataset}.txt

    python -m pyserini.eval.trec_eval \
    -c -m recall.100 beir-v1.0.0-${dataset}-test \
    run.beir.${model_name}.${dataset}.txt

    rm -r indices/models_${model_name}_${dataset}_index
done