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
        df = df.drop(columns=['systemBoundaryCompleteness.@type', 'systemBoundaryCompleteness.updatedVersion',
                              "product_properties.0.term.@type",
                              "product_properties.0.term.termType", "product_properties.0.term.@id",
                              "product_properties.0.@type",
                              "product_properties.1.term.@type", "product_properties.1.min", "product_properties.1.sd",
                              "product_properties.1.term.termType", "product_properties.1.term.@id",
                              "product_properties.2.term.@id",
                              "product_properties.1.@type", "product_properties", "product_properties.0.date",
                              "product_properties.2.@type",
                              "product_properties.0.min", "product_properties.0.max",
                              "product_properties.2.methodClassification",
                              "product_properties.2.methodClassificationDescription",
                              "product_properties.0.statsDefinition",
                              "product_properties.1.max", "product_properties.1.statsDefinition",
                              "product_properties.0.sd", "product_properties.1.methodClassification",
                              "product_properties.1.methodClassificationDescription",
                              "product_properties.0.methodClassification",
                              "product_properties.0.methodClassificationDescription", "product_properties.3.term.@type",
                              "product_properties.3.term.termType",
                              "product_properties.3.term.@id", "product_properties.3.@type",
                              "product_properties.0.methodModelDescription",
                              "product_properties.1.methodModelDescription", "product_properties.2.term.@type",
                              "product_properties.2.term.termType"
                              ])
    else:
        df = df.drop(columns=['systemBoundaryCompleteness.@type', "product_properties.0.term.@type",
                              "product_properties.0.term.termType", "product_properties.0.term.@id",
                              "product_properties.0.@type",
                              "product_properties.1.term.@type", "product_properties.1.min", "product_properties.1.sd",
                              "product_properties.1.term.termType", "product_properties.1.term.@id",
                              "product_properties.2.term.@id",
                              "product_properties.1.@type", "product_properties", "product_properties.0.date",
                              "product_properties.2.@type",
                              "product_properties.0.min", "product_properties.0.max",
                              "product_properties.2.methodClassification",
                              "product_properties.2.methodClassificationDescription",
                              "product_properties.0.statsDefinition",
                              "product_properties.1.max", "product_properties.1.statsDefinition",
                              "product_properties.0.sd", "product_properties.1.methodClassification",
                              "product_properties.1.methodClassificationDescription",
                              "product_properties.0.methodClassification",
                              "product_properties.0.methodClassificationDescription", "product_properties.3.term.@type",
                              "product_properties.3.term.termType",
                              "product_properties.3.term.@id", "product_properties.3.@type",
                              "product_properties.0.methodModelDescription",
                              "product_properties.1.methodModelDescription", "product_properties.2.term.@type",
                              "product_properties.2.term.termType"
                              ])

    # relabel columns - Hestia says FU are given by the term of the product (units)
    df.columns = df.columns.str.replace('units', 'functionalUnit')

    # write out file
    df.to_csv(directory + "input_data.csv", index=False)


if __name__ == "__main__":
    main()
