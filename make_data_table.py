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

    # drop rows with non-livestock products
    print(df['IA_productName'].unique())
    to_drop = ['Deer', 'Domestic duck', 'Electricity, produced on-site, solar PV', 'Meat, deer (cold carcass weight)',
               'Meat, deer (liveweight)', 'Meat, freshwater snails (with shell)', 'Meat, game (cold carcass weight)',
               'Meat, oyster (without shell)', 'Meat, rabbit (cold carcass weight)', 'Offal, deer', 'Rabbit',
               'Rabbit, doe', 'Rabbit, kit (weaned)', 'Rice, grain (in husk), flooded', 'Shell, freshwater snails',
               'Shell, oyster', 'Snail meal, without shells', 'Wastewater (kg mass)', 'Wheat, grain']

    # list of products that are not livestock or livestock adjacent
    print('Number of records dropped: ' + str(len(df[df['IA_productName'].isin(to_drop)])))
    df = df[~df['IA_productName'].isin(to_drop)]

    # relabel columns - Hestia says FU are given by the term of the product (units)
    df.columns = df.columns.str.replace('units', 'functionalUnit')

    # write out file
    df.to_csv(directory + "input_data.csv", index=False)


if __name__ == "__main__":
    prefix = "./data/hestia/"  # "llm-goal-scope/data/hestia/" on AWS
    main(prefix)
    main(prefix + "recalculated/")
