import data_cleaning
import make_data_table
import qa_dataset_creation
from multiprocessing import freeze_support


if __name__ == '__main__':
    freeze_support()
    data_cleaning.main("./data/")
    make_data_table.main("./data/")
    qa_dataset_creation.main("./data/", True)
    qa_dataset_creation.main("./data/", False)

    data_cleaning.main("./data/recalculated/")
    make_data_table.main("./data/recalculated/")
    qa_dataset_creation.main("./data/recalculated/", True)
    qa_dataset_creation.main("./data/recalculated/", False)
