dataset=msmarco
for model_name in 'msmarco_bge_embedding_model'; do
    python3 encode_corpus.py --model_name models/${model_name} --normalize --pooling cls --batch_size 1800 --dataset ${dataset}

    python -m pyserini.search.faiss \
    --threads 16 --batch-size 512 \
    --encoder-class auto \
    --encoder models/${model_name} --l2-norm --query-prefix "Represent this sentence for searching relevant passages: " \
    --index indices/models_${model_name}_${dataset}_index \
    --topics dl19-passage \
    --output run.${model_name}.dl19.txt \
    --hits 1000

    python -m pyserini.eval.trec_eval -c -l 2 -m map dl19-passage \
    run.${model_name}.dl19.txt 
    python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 dl19-passage \
    run.${model_name}.dl19.txt
    python -m pyserini.eval.trec_eval -c -l 2 -m recall.100 dl19-passage \
    run.${model_name}.dl19.txt  
    python -m pyserini.eval.trec_eval -c -l 2 -m recall.1000 dl19-passage \
    run.${model_name}.dl19.txt 

    python -m pyserini.search.faiss \
    --threads 16 --batch-size 512 \
    --encoder-class auto \
    --encoder models/${model_name} --l2-norm  --query-prefix "Represent this sentence for searching relevant passages:" \
    --index indices/models_${model_name}_${dataset}_index \
    --topics dl20 \
    --output run.${model_name}.dl20.txt  \
    --hits 1000

    python -m pyserini.eval.trec_eval -c -l 2 -m map dl20-passage \
    run.${model_name}.dl20.txt 
    python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 dl20-passage \
    run.${model_name}.dl20.txt
    python -m pyserini.eval.trec_eval -c -l 2 -m recall.100 dl20-passage \
    run.${model_name}.dl20.txt  
    python -m pyserini.eval.trec_eval -c -l 2 -m recall.1000 dl20-passage \
    run.${model_name}.dl20.txt 

    python -m pyserini.search.faiss \
    --threads 16 --batch-size 512 \
    --encoder-class auto \
    --encoder models/${model_name} --l2-norm  --query-prefix "Represent this sentence for searching relevant passages:" \
    --index indices/models_${model_name}_${dataset}_index  \
    --topics msmarco-passage-dev-subset \
    --output run.${model_name}.dev.txt  \
    --hits 1000


  python -m pyserini.eval.trec_eval -c -M 10 -m recip_rank msmarco-passage-dev-subset \
    run.${model_name}.dev.txt
  python -m pyserini.eval.trec_eval -c -m recall.100 msmarco-passage-dev-subset \
    run.${model_name}.dev.txt

  rm -r indices/models_${model_name}_${dataset}_index
done


for model_name in 'msmarco_snowflake_embedding_model'; do
    python3 encode_corpus.py --model_name models/${model_name} --normalize --pooling cls --batch_size 1800 --dataset ${dataset}

    python -m pyserini.search.faiss \
    --threads 16 --batch-size 512 \
    --encoder-class auto \
    --encoder models/${model_name} --l2-norm --query-prefix "Represent this sentence for searching relevant passages: " \
    --index indices/models_${model_name}_${dataset}_index \
    --topics dl19-passage \
    --output run.${model_name}.dl19.txt \
    --hits 1000

    python -m pyserini.eval.trec_eval -c -l 2 -m map dl19-passage \
    run.${model_name}.dl19.txt 
    python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 dl19-passage \
    run.${model_name}.dl19.txt
    python -m pyserini.eval.trec_eval -c -l 2 -m recall.100 dl19-passage \
    run.${model_name}.dl19.txt  
    python -m pyserini.eval.trec_eval -c -l 2 -m recall.1000 dl19-passage \
    run.${model_name}.dl19.txt 

    python -m pyserini.search.faiss \
    --threads 16 --batch-size 512 \
    --encoder-class auto \
    --encoder models/${model_name} --l2-norm  --query-prefix "Represent this sentence for searching relevant passages:" \
    --index indices/models_${model_name}_${dataset}_index \
    --topics dl20 \
    --output run.${model_name}.dl20.txt  \
    --hits 1000

    python -m pyserini.eval.trec_eval -c -l 2 -m map dl20-passage \
    run.${model_name}.dl20.txt 
    python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 dl20-passage \
    run.${model_name}.dl20.txt
    python -m pyserini.eval.trec_eval -c -l 2 -m recall.100 dl20-passage \
    run.${model_name}.dl20.txt  
    python -m pyserini.eval.trec_eval -c -l 2 -m recall.1000 dl20-passage \
    run.${model_name}.dl20.txt 

    python -m pyserini.search.faiss \
    --threads 16 --batch-size 512 \
    --encoder-class auto \
    --encoder models/${model_name} --l2-norm  --query-prefix "Represent this sentence for searching relevant passages:" \
    --index indices/models_${model_name}_${dataset}_index  \
    --topics msmarco-passage-dev-subset \
    --output run.${model_name}.dev.txt  \
    --hits 1000


  python -m pyserini.eval.trec_eval -c -M 10 -m recip_rank msmarco-passage-dev-subset \
    run.${model_name}.dev.txt
  python -m pyserini.eval.trec_eval -c -m recall.100 msmarco-passage-dev-subset \
    run.${model_name}.dev.txt

  rm -r indices/models_${model_name}_${dataset}_index
done


for model_name in 'msmarco_gte_embedding_model_contrastive'; do
    python3 encode_corpus.py --model_name models/${model_name} --normalize --pooling cls --batch_size 1800 --dataset ${dataset}

    python -m pyserini.search.faiss \
    --threads 16 --batch-size 512 \
    --encoder-class auto \
    --encoder models/${model_name} --l2-norm --query-prefix "" \
    --index indices/models_${model_name}_${dataset}_index \
    --topics dl19-passage \
    --output run.${model_name}.dl19.txt \
    --hits 1000

    python -m pyserini.eval.trec_eval -c -l 2 -m map dl19-passage \
    run.${model_name}.dl19.txt 
    python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 dl19-passage \
    run.${model_name}.dl19.txt
    python -m pyserini.eval.trec_eval -c -l 2 -m recall.100 dl19-passage \
    run.${model_name}.dl19.txt  
    python -m pyserini.eval.trec_eval -c -l 2 -m recall.1000 dl19-passage \
    run.${model_name}.dl19.txt 

    python -m pyserini.search.faiss \
    --threads 16 --batch-size 512 \
    --encoder-class auto \
    --encoder models/${model_name} --l2-norm  --query-prefix "" \
    --index indices/models_${model_name}_${dataset}_index \
    --topics dl20 \
    --output run.${model_name}.dl20.txt  \
    --hits 1000

    python -m pyserini.eval.trec_eval -c -l 2 -m map dl20-passage \
    run.${model_name}.dl20.txt 
    python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 dl20-passage \
    run.${model_name}.dl20.txt
    python -m pyserini.eval.trec_eval -c -l 2 -m recall.100 dl20-passage \
    run.${model_name}.dl20.txt  
    python -m pyserini.eval.trec_eval -c -l 2 -m recall.1000 dl20-passage \
    run.${model_name}.dl20.txt 


    python -m pyserini.search.faiss \
    --threads 16 --batch-size 512 \
    --encoder-class auto \
    --encoder models/${model_name} --l2-norm  --query-prefix "" \
    --index indices/models_${model_name}_${dataset}_index  \
    --topics msmarco-passage-dev-subset \
    --output run.${model_name}.dev.txt  \
    --hits 1000


  python -m pyserini.eval.trec_eval -c -M 10 -m recip_rank msmarco-passage-dev-subset \
    run.${model_name}.dev.txt
  python -m pyserini.eval.trec_eval -c -m recall.100 msmarco-passage-dev-subset \
    run.${model_name}.dev.txt

    rm -r indices/models_${model_name}_${dataset}_index
done


for model_name in 'msmarco_e5_embedding_model'; do
    python3 encode_corpus_e5.py --model_name models/${model_name} --normalize --pooling mean --batch_size 1800 --dataset ${dataset}

    python -m pyserini.search.faiss \
    --threads 16 --batch-size 512 \
    --encoder-class auto \
    --encoder models/${model_name} --pooling mean --l2-norm --query-prefix "query: " \
    --index indices/models_${model_name}_${dataset}_index \
    --topics dl19-passage \
    --output run.${model_name}.dl19.txt \
    --hits 1000

    python -m pyserini.eval.trec_eval -c -l 2 -m map dl19-passage \
    run.${model_name}.dl19.txt 
    python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 dl19-passage \
    run.${model_name}.dl19.txt
    python -m pyserini.eval.trec_eval -c -l 2 -m recall.100 dl19-passage \
    run.${model_name}.dl19.txt  
    python -m pyserini.eval.trec_eval -c -l 2 -m recall.1000 dl19-passage \
    run.${model_name}.dl19.txt 

    python -m pyserini.search.faiss \
    --threads 16 --batch-size 512 \
    --encoder-class auto \
    --encoder models/${model_name} --pooling mean --l2-norm  --query-prefix "query: " \
    --index indices/models_${model_name}_${dataset}_index \
    --topics dl20 \
    --output run.${model_name}.dl20.txt  \
    --hits 1000

    python -m pyserini.eval.trec_eval -c -l 2 -m map dl20-passage \
    run.${model_name}.dl20.txt 
    python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 dl20-passage \
    run.${model_name}.dl20.txt
    python -m pyserini.eval.trec_eval -c -l 2 -m recall.100 dl20-passage \
    run.${model_name}.dl20.txt  
    python -m pyserini.eval.trec_eval -c -l 2 -m recall.1000 dl20-passage \
    run.${model_name}.dl20.txt 


    python -m pyserini.search.faiss \
    --threads 16 --batch-size 512 \
    --encoder-class auto \
    --encoder models/${model_name} --l2-norm --pooling mean --query-prefix "query: " \
    --index indices/models_${model_name}_${dataset}_index  \
    --topics msmarco-passage-dev-subset \
    --output run.${model_name}.dev.txt  \
    --hits 1000


  python -m pyserini.eval.trec_eval -c -M 10 -m recip_rank msmarco-passage-dev-subset \
    run.${model_name}.dev.txt
  python -m pyserini.eval.trec_eval -c -m recall.100 msmarco-passage-dev-subset \
    run.${model_name}.dev.txt

    rm -r indices/models_${model_name}_${dataset}_index
done