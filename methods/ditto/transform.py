# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import sys

import pandas as pd
import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split


NUM_NEG_PAIRS = 10000


def generate_candidates(tableA_df, tableB_df, matches_df, seed):
    # rename columns of tableA and tableB for an easier joining of DataFrames
    cand_tableA = tableA_df.add_prefix('tableA_')
    cand_tableB = tableB_df.add_prefix('tableB_')

    # create table of matching pairs, which contains all attributes
    pos_pairs = pd.concat([
        (cand_tableA.iloc[matches_df['tableA_id']]).reset_index(drop=True),
        (cand_tableB.iloc[matches_df['tableB_id']]).reset_index(drop=True)
    ], axis=1)

    assert matches_df.equals(pos_pairs.loc[:, ['tableA_id', 'tableB_id']]), \
        "Positive pair creation failed, pairs not identical with matches.csv"

    # create table of (randomly sampled) non-matching pairs, again containing all necessary attributes
    matches_tuples = set(zip(*map(matches_df.get, matches_df)))

    if NUM_NEG_PAIRS == -1:
        full_pairs = set(tuple(pair for pair in product(tableA_df['id'], tableB_df['id'])))
        neg_tuples = full_pairs - matches_tuples
        neg_ids = np.array(list(neg_tuples))
    else:
        rng = np.random.default_rng(seed)
        num_neg_pairs = 0
        skip_counter = 0
        neg_ids = []
        while num_neg_pairs < NUM_NEG_PAIRS:
            assert skip_counter < NUM_NEG_PAIRS * 1.5, "Too many pairs skipped, please check the number of negatives requested"
            a_id = rng.integers(0, tableA_df.shape[0])
            b_id = rng.integers(0, tableB_df.shape[0])

            if (a_id, b_id) in matches_tuples:
                skip_counter += 1
                continue
            neg_ids.append([a_id, b_id])
            num_neg_pairs += 1
        neg_ids = np.array(neg_ids)

    neg_pairs = pd.concat([
        (cand_tableA.iloc[neg_ids[:, 0]]).reset_index(drop=True),
        (cand_tableB.iloc[neg_ids[:, 1]]).reset_index(drop=True)
    ], axis=1)

    pos_pairs['label'] = 1
    neg_pairs['label'] = 0

    # join the matching and non-matching pairs to a large table
    pairs = pd.concat([pos_pairs, neg_pairs]).reset_index(drop=True)
    pairs['_id'] = np.arange(pairs.shape[0])
    return pairs

def join_columns (table, columns_to_join=None, separator=' ', prefixes=['tableA_', 'tableB_']):
    agg_table = pd.DataFrame()
    for prefix in prefixes:

        if columns_to_join == None:
            columns = [column for column in table.columns if (column != prefix+'id' and prefix in column)]
        else:
            columns = [prefix+column for column in columns_to_join]

        
        red_table = table.loc[:,columns]
        red_table = red_table.fillna('')
        red_table = red_table.astype(str)
        
        for column in columns:
            red_table[column] = f"COL {column.replace(prefix, '')} VAL " + red_table[column] 
        
        part_table = red_table.aggregate(separator.join, axis=1)
        #part_table = part_table.map(lambda x: x.replace('nan', ''))
        part_table.rename(prefix+'AgValue', inplace=True)
        
        agg_table = pd.concat([agg_table,part_table], axis=1)
    
    return pd.concat([agg_table, table['label']], axis=1),\
        pd.concat([table[prefixes[0] + 'id'], table[prefixes[1] + 'id']], axis=1)

def transform_input(source_dir, output_dir, columns_to_join=None, separator=' ', prefixes=['tableA_', 'tableB_']):
    train_df = pd.read_csv(os.path.join(source_dir, 'train.csv'), encoding_errors='replace')
    test_df = pd.read_csv(os.path.join(source_dir, 'test.csv'), encoding_errors='replace')
    
    train, train_id = join_columns(train_df, columns_to_join, separator, prefixes)
    test, test_id = join_columns(test_df, columns_to_join, separator, prefixes)
    
    train_file = os.path.join(output_dir, 'train.tsv')
    test_file = os.path.join(output_dir, 'test.tsv')
    train.to_csv(train_file, '\t', header=False, index=False)
    test.to_csv(test_file, '\t', header=False, index=False)
    return train_file, test_file, train_id, test_id



def transform_input_old(source_dir, output_dir, recall, seed):
    """
    The source directory contains the following files (in common format for all methods):
     - tableA.csv (where the first row is the header, and it has to contain the id attribute)
     - tableB.csv (same as tableA.csv)
     - matches.csv (should have tableA_id, tableB_id attributes, which means that the tableA_id record is a match with the tableB_id record)

    The output directory should contain the files converted into the format expected by the method.
    """

    tableA_df = pd.read_csv(os.path.join(source_dir, 'tableA.csv'), encoding_errors='replace')
    tableB_df = pd.read_csv(os.path.join(source_dir, 'tableB.csv'), encoding_errors='replace')
    matches_df = pd.read_csv(os.path.join(source_dir, 'matches.csv'), encoding_errors='replace')
    # TODO: reduce GT based on recall value

    pairs = generate_candidates(tableA_df, tableB_df, matches_df, seed)
    train_df, test_df = train_test_split(pairs, train_size=0.7, random_state=seed, shuffle=True, stratify=pairs['label'])
    train, train_id = join_columns(train_df)
    test, test_id = join_columns(test_df)
    
    train_file = os.path.join(output_dir, 'train.tsv')
    test_file = os.path.join(output_dir, 'test.tsv')
    train.to_csv(train_file, '\t', header=False, index=False)
    test.to_csv(test_file, '\t', header=False, index=False)
    return train_file, test_file, train_id, test_id


def transform_output(predictions, ids, labels, runtime, dest_dir):
    """
    Transform the output of the method into two common format files, which are stored in the destination directory.
    metrics.csv: f1, precision, recall, time (1 row, 4 columns, with header)
    predictions.csv: tableA_id, tableB_id, etc. (should have at least 2 columns and a header row)

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Output of the Matcher contains the match/non-match prediction and the
        true label for each pair in the test set.
        Has at least the columns tableA_id, tableB_id, prediction, label.
    runtime : float
        Measured runtime of the matcher in seconds.
    dest_dir : str
        Directory name where the output should be stored.
    """

    # get the actual candidates (entity pairs with prediction 1)
    predictions_df = pd.DataFrame({'prediction':predictions, 'tableA_id':ids['tableA_id'], 'tableB_id':ids['tableB_id'], 'label':labels})
    #predictions_df = pd.concat([predictions, ids, labels], axis=1)
    #predictions_df.columns = ['prediction', 'tableA_id', 'tableB_id', 'label']
    print(predictions_df)
    
    
    candidate_table = predictions_df[predictions_df['prediction'] == 1]
    print(candidate_table)
    # save candidate pair IDs to predictions.csv
    candidate_table[['tableA_id', 'tableB_id']].to_csv(os.path.join(dest_dir, 'predictions.csv'), index=False)

    # calculate evaluation metrics
    num_candidates = candidate_table.shape[0]
    true_positives = candidate_table['label'].sum()
    ground_truth = predictions_df['label'].sum()

    recall = true_positives / ground_truth
    precision = true_positives / num_candidates
    f1 = 2 * precision * recall / (precision + recall)

    # save evaluation metrics to metrics.csv
    pd.DataFrame({
        'f1': [f1],
        'precision': [true_positives / num_candidates],
        'recall': [true_positives / ground_truth],
        'time': [runtime],
    }).to_csv(os.path.join(dest_dir, 'metrics.csv'), index=False)
    return None


if __name__ == "__main__":
    in_path = sys.argv[1]
    out_path = sys.argv[2]

    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    transform_input(in_path)