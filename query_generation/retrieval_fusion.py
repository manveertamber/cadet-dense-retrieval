# python script used to fuse retrieval results from three retrievers to get a single passage set for reranking

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os

def parse_retrieval_file(file_path):
    query_rankings = defaultdict(list)
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            query = parts[0]
            passage = parts[2]
            rank = int(parts[3])
            query_rankings[query].append((passage, rank))
    return query_rankings

def reciprocal_rank_fusion(query_rankings_list, K=60):
    rrf_scores = defaultdict(float)
    for query_rankings in query_rankings_list:
        for passage, rank in query_rankings:
            rrf_scores[passage] += 1 / (rank + K)
    
    sorted_passages = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(passage, score) for passage, score in sorted_passages]

def fuse_retrieval_files(file_paths, output_file, K=60):
    all_query_rankings = defaultdict(list)
    for file_path in file_paths:
        query_rankings = parse_retrieval_file(file_path)
        for query, rankings in query_rankings.items():
            all_query_rankings[query].append(rankings)
    
    with open(output_file, 'w') as out_file:
        with ThreadPoolExecutor() as executor:
            futures = {}
            with tqdm(total=len(all_query_rankings), desc="Fusing queries") as pbar:
                for query, query_rankings_list in all_query_rankings.items():
                    futures[executor.submit(reciprocal_rank_fusion, query_rankings_list, K)] = query
                
                # Write each result as it completes
                for future in as_completed(futures):
                    query = futures[future]
                    fused_results = future.result()
                    for rank, (passage, score) in enumerate(fused_results, 1):
                        out_file.write(f"{query} Q0 {passage} {rank} {score:.6f} RRF\n")
                    pbar.update(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuse retrieval files using Reciprocal Rank Fusion")
    parser.add_argument('--runs', nargs='+', required=True, help="List of retrieval file paths to be fused")
    parser.add_argument('--output', required=True, help="Output file path for the fused results")
    args = parser.parse_args()

    fuse_retrieval_files(args.runs, args.output)
