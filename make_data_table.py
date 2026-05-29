import os
import pandas as pd
import json


def main(directory):
    """
    Remove some of the unnecessary input data from HESTIA
    :param directory: input directory of HESTIA data
    :return: cleaned input.csv files for use in building the json.ld dataset
    """
    # get the directory
    directory_path = directory + "cleaned/"
    df = pd.DataFrame()

    # for each file in the directory, iterate through and add to big table
    for entry_name in os.listdir(directory_path):
        with open(directory_path + entry_name, 'r', encoding='utf-8') as f:
            data = json.load(f)

            # normalize data (completeness)
            table_data = pd.json_normalize(data)

        df = pd.concat([df, table_data])

    # drop unnecessary columns
    if "recalculated" in directory:
        df = df.drop(columns=['systemBoundaryCompleteness.@type',  # no real data
                              'systemBoundaryCompleteness.updated',   # unnecessary, what was recalcualted compared to original
                              'systemBoundaryCompleteness.updatedVersion',  # unnecessary, what was recalcualted compared to original
                              'IAmethodClassification',  # mostly empty
                              'IAproductFate',  # mostly empty
                              'IAfunctionalUnitQuantity',  # all 1
                              'siteMethodClassification'  # mostly empty
                              ])
    else:
        df = df.drop(columns=['systemBoundaryCompleteness.@type',  # no data
                              'IAmethodClassification', # mostly empty
                              'IAproductFate',  # mostly empty
                              'IAfunctionalUnitQuantity',  # all 1
                              'siteMethodClassification'  # mostly empty
                              ])

    # relabel columns - Hestia says FU are given by the term of the product (units)
    df.columns = df.columns.str.replace('units', 'functionalUnit')

    # write out file
    df.to_csv(directory + "input_data.csv", index=False)


if __name__ == "__main__":
    prefix = "./data/hestia/"  # "llm-goal-scope/data/hestia/" on AWS
    main(prefix)
    main(prefix + "recalculated/")
