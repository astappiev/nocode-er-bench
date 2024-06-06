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
        neg_ids = set([])
        while num_neg_pairs < NUM_NEG_PAIRS:
            assert skip_counter < NUM_NEG_PAIRS * 1.5, "Too many pairs skipped, please check the number of negatives requested"
            a_id = rng.integers(0, tableA_df.shape[0])
            b_id = rng.integers(0, tableB_df.shape[0])

            if (a_id, b_id) in matches_tuples or (a_id, b_id) in neg_ids:
                skip_counter += 1
                continue
            neg_ids.add((a_id, b_id))
            num_neg_pairs += 1
        neg_ids = np.array(list(neg_ids))

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


def transform_input(source_dir, recall, seed):
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
    train, test = train_test_split(pairs, train_size=recall, random_state=seed, shuffle=True, stratify=pairs['label'])
    return tableA_df, tableB_df, train, test


def transform_output(predictions_df, runtime, dest_dir):
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
    candidate_table = predictions_df[predictions_df['prediction'] == 1]
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
