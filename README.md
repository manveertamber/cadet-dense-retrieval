# CADET Embeddings

We present CADET, a framework for fine-tuning embedding models for retrieval on specific corpora using diverse synthetic queries and cross-encoder listwise distillation.

We will continue to refine this codebase. For questions or support, please reach out to [mtamber@uwaterloo.ca](mailto:mtamber@uwaterloo.ca).

**Model link:** [cadet-embed-base-v1 on Hugging Face](https://huggingface.co/manveertamber/cadet-embed-base-v1)

## Overview

### Directories

- **encoding/**  
  Contains scripts to encode corpora and evaluate models.

- **query_generation/**  
  Includes scripts for generating synthetic queries.

- **reranker/**  
  Code for reranking.

- **training_scripts/**  
  Scripts for fine-tuning models.
---

If you use CADET, please cite the following paper: 
```
  @article{tamber2025teaching,
    title={Teaching Dense Retrieval Models to Specialize with Listwise Distillation and LLM Data Augmentation},
    author={Tamber, Manveer Singh and Kazi, Suleman and Sourabh, Vivek and Lin, Jimmy},
    journal={arXiv:2502.19712},
    year={2025}
  }
```