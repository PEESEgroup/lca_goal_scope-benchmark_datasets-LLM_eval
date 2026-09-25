import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from absl import logging  # really only needed on AWS to quiet logging purposes

logging.set_verbosity(logging.ERROR)
import rag
import data_cleaning
import make_data_table
import qa_dataset_creation
import evaluate_models
import covariance
import data_analysis
from multiprocessing import freeze_support

if __name__ == '__main__':
    # set up RAG vector database
    rag.main()

    # main method to clean data, remove excessive data, and build the json-ld dataset
    # also attempts to quiet some of the excessive logging in AWS - unsure to what extent that happens
    freeze_support()
    data_cleaning.main("llm-goal-scope/data/")
    make_data_table.main("llm-goal-scope/data/")
    qa_dataset_creation.main("llm-goal-scope/data/qa_dataset/original/no_rag/", "llm-goal-scope/data/", False)
    qa_dataset_creation.main("llm-goal-scope/data/qa_dataset/original/rag/", "llm-goal-scope/data/", True)
    data_cleaning.main("llm-goal-scope/data/recalculated/")
    make_data_table.main("llm-goal-scope/data/recalculated/")
    qa_dataset_creation.main("llm-goal-scope/data/qa_dataset/recalculated/rag/", "llm-goal-scope/data/recalculated/",
                             True)
    qa_dataset_creation.main("llm-goal-scope/data/qa_dataset/recalculated/no_rag/", "llm-goal-scope/data/recalculated/",
                             False)

    # evaluate the models on the dataset
    evaluate_models.main()

    # analyze model data
    data_analysis.main()

    # produce other graphs about the dataset and not the encoder-classifier model results
    covariance.main()
