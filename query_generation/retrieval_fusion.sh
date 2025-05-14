#!/bin/bash

# bash script used to fuse retrieval results from three retrievers to get a single passage set for reranking

run_files_directory="retrieval_runs"

models=("Alibaba-NLP_gte-base-en-v1.5" "Snowflake_snowflake-arctic-embed-m-v1.5" "BAAI_bge-base-en-v1.5")
datasets=("msmarco" "fiqa" "scifact" "trec-covid" "nfcorpus" "arguana" "webis-touche2020" "dbpedia-entity" "scidocs" "climate-fever")
query_types=("titles" "claims" "questions" "random" "msmarco" "keywords")

for dataset in "${datasets[@]}"; do
  for query_type in "${query_types[@]}"; do
    
    matching_files=()
    for model in "${models[@]}"; do
      file="$run_files_directory/run.${model}.${dataset}.generated-queries-${query_type}.filtered_20.txt"
      if [ -f "$file" ]; then
        matching_files+=("$file")
      fi
    done
    
    if [ ${#matching_files[@]} -eq 3 ]; then
      output_file="$run_files_directory/run.hybrid.${dataset}.generated-queries-${query_type}.filtered.txt"
      
      echo "Command: python -m pyserini.fusion --runs ${matching_files[*]} --output $output_file"
      
      python3 fuse.py --runs "${matching_files[@]}" --output "$output_file"
      
      echo "Output saved to $output_file"
    else
      echo "Skipping $dataset, $query_type: not all three model files are available."
    fi
    
  done
done
