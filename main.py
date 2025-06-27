import data_cleaning
import make_data_table

data_cleaning.main("./data/")
make_data_table.main("./data/")

data_cleaning.main("./data/recalculated/")
make_data_table.main("./data/recalculated/")
