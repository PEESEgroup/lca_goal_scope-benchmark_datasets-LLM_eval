import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from absl import logging
logging.set_verbosity(logging.ERROR)
import data_cleaning
import make_data_table
import qa_dataset_creation
from multiprocessing import freeze_support


if __name__ == '__main__':
    freeze_support()
    data_cleaning.main("llm-goal-scope/data/")
    make_data_table.main("llm-goal-scope/data/")
    qa_dataset_creation.main("llm-goal-scope/data/qa_dataset/original/no_rag/", "llm-goal-scope/data/",False)
    qa_dataset_creation.main("llm-goal-scope/data/qa_dataset/original/rag/", "llm-goal-scope/data/", True)

    data_cleaning.main("llm-goal-scope/data/recalculated/")
    make_data_table.main("llm-goal-scope/data/recalculated/")
    qa_dataset_creation.main("llm-goal-scope/data/qa_dataset/recalculated/rag/", "llm-goal-scope/data/recalculated/", True)
    qa_dataset_creation.main("llm-goal-scope/data/qa_dataset/recalculated/no_rag/", "llm-goal-scope/data/recalculated/",False)

