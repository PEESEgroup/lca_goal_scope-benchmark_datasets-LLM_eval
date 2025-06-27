import os
import pandas as pd
import json


def main():
    # get the directory
    directory_path = "./data/cleaned/"
    df = pd.DataFrame()

    # for each file in the directory, iterate through and add to big table
    for entry_name in os.listdir(directory_path):
        with open(directory_path + entry_name, 'r', encoding='utf-8') as f:
            data = json.load(f)

            # normalize data (completeness)
            table_data = pd.json_normalize(data)
            test = pd.json_normalize(table_data["product_properties"])

            # drop unnecessary completeness columns
        df = pd.concat([df, table_data])

    # write out file
    df.to_csv("data/input_data.csv", index=False)


if __name__ == "__main__":
    main()