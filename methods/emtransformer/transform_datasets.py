import csv
import logging
import os
from typing import Callable

import pandas as pd

import constants as C
from logging_customized import setup_logging

setup_logging()


def convert_deepmatcher_structure(filename_matches: str, filename_table_a: str,
                                  filename_table_b: str, filename_destination: str,
                                  fun_join_left: Callable, fun_join_right: Callable):
    df_table_a = pd.read_csv(filename_table_a, sep=",", encoding="UTF-8")
    df_table_b = pd.read_csv(filename_table_b, sep=",", encoding="UTF-8")

    with open(filename_matches) as src, open(filename_destination, "w+") as dest:
        reader = csv.DictReader(src, delimiter=",")
        writer = csv.writer(dest, delimiter="\t")

        writer.writerow([C.INDEX_KEY, C.TEXT_LEFT, C.TEXT_RIGHT, C.LABEL])

        for idx, line in enumerate(reader):
            # find the row in tableA and tableB which contain the actual values. Make sure both keys have the same data type (we use string comparison as keys can be strings)
            # As there should only exist one row per id, we use iloc (which returns a data series)
            left = df_table_a.loc[df_table_a['id'].astype(str) == line['ltable_id']].iloc[0]
            right = df_table_b.loc[df_table_b['id'].astype(str) == line['rtable_id']].iloc[0]

            # which attributes we join is depending on the dataset
            left_joined_attributes = fun_join_left(left)
            right_joined_attributes = fun_join_right(right)

            writer.writerow([idx, left_joined_attributes, right_joined_attributes, line['label']])

    logging.info("Created new tsv. Filename: {}".format(filename_destination))


def abt_buy_join_left(row: pd.Series):
    # return " ".join([left['name'], left['description']])
    return row['description']


def abt_buy_join_right(row: pd.Series):
    if row.isnull()['description']:
        right_joined_attributes = row['name']
    else:
        right_joined_attributes = " ".join([row['name'], row['description']])

    return right_joined_attributes


def company_join(row: pd.Series):
    return row['content']


def full_join_except_id(row: pd.Series):
    single_text_blob = []
    for index, row_value in row[1:].items():
        if not row.isnull()[index]:
            single_text_blob.append(str(row_value))  # make sure non-string values (e.g. prices as floats) get converted

    return " ".join(single_text_blob)


if __name__ == "__main__":

    # Source to destination mapping
    dataset_dir = [
      ('../existing_datasets/Structured/DBLP-ACM/', 'Ds1/'),
      ('../existing_datasets/Structured/DBLP-GoogleScholar/', 'Ds2/'),
      ('../existing_datasets/Structured/iTunes-Amazon/', 'Ds3/'),
      ('../existing_datasets/Structured/Walmart-Amazon/', 'Ds4/'),
      ('../existing_datasets/Structured/Beer/', 'Ds5/'),
      ('../existing_datasets/Structured/Amazon-Google/', 'Ds6/'),
      ('../existing_datasets/Structured/Fodors-Zagats/', 'Ds7/'),
      ('../existing_datasets/Dirty/DBLP-ACM/', 'Dd1/'),
      ('../existing_datasets/Dirty/DBLP-GoogleScholar/', 'Dd2/'),
      ('../existing_datasets/Dirty/iTunes-Amazon/', 'Dd3/'),
      ('../existing_datasets/Dirty/Walmart-Amazon/', 'Dd4/'),
      ('../existing_datasets/Textual/Abt-Buy/', 'Dt1/'),
      ('../existing_datasets/Textual/Company/', 'Dt2/')
    ]

    for source_dir, dest_dir in dataset_dir:

      if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)

      if 'Abt-Buy' in source_dir:
          convert_deepmatcher_structure(os.path.join(source_dir, "train.csv"),
                                        os.path.join(source_dir, "tableA.csv"),
                                        os.path.join(source_dir, "tableB.csv"),
                                        os.path.join(dest_dir, "train.tsv"),
                                        abt_buy_join_left, abt_buy_join_right)

          convert_deepmatcher_structure(os.path.join(source_dir, "test.csv"),
                                        os.path.join(source_dir, "tableA.csv"),
                                        os.path.join(source_dir, "tableB.csv"),
                                        os.path.join(dest_dir, "test.tsv"),
                                        abt_buy_join_left, abt_buy_join_right)

          convert_deepmatcher_structure(os.path.join(source_dir, "val.csv"),
                                        os.path.join(source_dir, "tableA.csv"),
                                        os.path.join(source_dir, "tableB.csv"),
                                        os.path.join(dest_dir, "dev.tsv"),
                                        abt_buy_join_left, abt_buy_join_right)

      elif 'Company' in source_dir:
          convert_deepmatcher_structure(os.path.join(source_dir, "train.csv"),
                                        os.path.join(source_dir, "tableA.csv"),
                                        os.path.join(source_dir, "tableB.csv"),
                                        os.path.join(dest_dir, "train.tsv"),
                                        company_join, company_join)

          convert_deepmatcher_structure(os.path.join(source_dir, "test.csv"),
                                        os.path.join(source_dir, "tableA.csv"),
                                        os.path.join(source_dir, "tableB.csv"),
                                        os.path.join(dest_dir, "test.tsv"),
                                        company_join, company_join)

          convert_deepmatcher_structure(os.path.join(source_dir, "val.csv"),
                                        os.path.join(source_dir, "tableA.csv"),
                                        os.path.join(source_dir, "tableB.csv"),
                                        os.path.join(dest_dir, "dev.tsv"),
                                        company_join, company_join)

      else:
          convert_deepmatcher_structure(os.path.join(source_dir, "train.csv"),
                                        os.path.join(source_dir, "tableA.csv"),
                                        os.path.join(source_dir, "tableB.csv"),
                                        os.path.join(dest_dir, "train.tsv"),
                                        full_join_except_id, full_join_except_id)

          convert_deepmatcher_structure(os.path.join(source_dir, "test.csv"),
                                        os.path.join(source_dir, "tableA.csv"),
                                        os.path.join(source_dir, "tableB.csv"),
                                        os.path.join(dest_dir, "test.tsv"),
                                        full_join_except_id, full_join_except_id)

          convert_deepmatcher_structure(os.path.join(source_dir, "val.csv"),
                                        os.path.join(source_dir, "tableA.csv"),
                                        os.path.join(source_dir, "tableB.csv"),
                                        os.path.join(dest_dir, "dev.tsv"),
                                        full_join_except_id, full_join_except_id)
