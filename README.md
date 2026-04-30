# Goal and Scope Datasets based on HESTIA data - Evaluated on Livestock LCA

This study creates Goal and Scope LLM benchmarks as a multi-label text classification task, then evaluates LLMs on this benchmark for livestock LCA. The main folders include
    data

## Files Overview

The root folder contains the Python scripts used in the study. There are eleven Python scripts and one folder. Details of each file or folder is provided below:

- data/: This folder contains four folders
  - hestia/: LCA data downloaded from the HESTIA platform
  - qa_dataset/ CONTAINS THE GOAL AND SCOPE DATASET BENCHMARKS. 
    - /original contains the original dataset
      - /no_rag without RAG
      - /rag with RAG
    - /recalculated contains standardized dataset
      - /no_rag without RAG
      - /rag with RAG
    - /results/ contains the results from benchmarking 7 LLMs on the 14 datasets
      - /qa_dataset_<dataset-name>\_<RAG boolean>\_RAG\_<LCA task>/<model-developer>/<model-name>/
        - contains confusion matrices, list of errors and predictions, loss functions, and the test metrics for the model on a particular dataset
      - scattered .csv and .png files containing summarized results
  - RAG-textract/: contains the RAG files used as input to the vector database
  - small_rag/: contains a few RAG files to test the workings of the vector database
- constants.py contains a list of constants for use in the project, including the names of tokenizing models
- covariance.py script for calculating the convariance and correlation between variables in the input dataset
- data_analysis.py produces many figures and tables for the mansucript
- data_cleaning.py cleans HESTIA data
- evaluate_models.py evaluate LLMs on the goal and scope benchmark dataset
- huggingface_login.py script to log in to huggingface
- main.py generate goal and scope datasets from HESTIA data
- make_data_table.py cleans HESTIA data
- qa_dataset_creation.py creates multi-label text classification datasets from cleaned HESTIA data 
- rag.py setup script for the vector database
- rag_retrieval.py retrieve relevant information from the vector database

## Requirements

To run the codes in this repository, python 3.11 was used and the package requirements can be found in requirements.txt:

## Citation

Please use the following citation when using the data, methods or results of this work:

TBD
