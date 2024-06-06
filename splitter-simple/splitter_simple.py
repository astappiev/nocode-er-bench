import argparse
from os import path
import random
import pathtype
import pandas as pd
import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split


def generate_candidates(tableA_df, tableB_df, matches_df, neg_pairs_limit=10000, seed=1):
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

    if neg_pairs_limit == -1:
        full_pairs = set(tuple(pair for pair in product(tableA_df['id'], tableB_df['id'])))
        neg_tuples = full_pairs - matches_tuples
        neg_ids = np.array(list(neg_tuples))
    else:
        rng = np.random.default_rng(seed)
        num_neg_pairs = 0
        skip_counter = 0
        neg_ids = set([])
        while num_neg_pairs < neg_pairs_limit:
            assert skip_counter < neg_pairs_limit * 1.5, \
                "Too many pairs skipped, please check the number of negatives requested"
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


def split_input(tableA_df, tableB_df, matches_df, recall=0.7, seed=1):
    candidates = generate_candidates(tableA_df, tableB_df, matches_df, seed=seed)
    print("Candidates generated: ", candidates.shape[0])
    return train_test_split(candidates, train_size=recall, random_state=seed, shuffle=True, stratify=candidates['label'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Splits the dataset using random method')
    parser.add_argument('input', type=pathtype.Path(readable=True), nargs='?', default='/data',
                        help='Input directory containing the dataset')
    parser.add_argument('output', type=pathtype.Path(writable=True), nargs='?',
                        help='Output directory to store the output')
    parser.add_argument('-r', '--recall', type=float, nargs='?', default=0.7,
                        help='The recall value for the train set')
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input

    print("Hi, I'm simple splitter, I'm doing random split of the input datasets into train and test sets")
    tableA_df = pd.read_csv(path.join(args.input, 'tableA.csv'), encoding_errors='replace')
    tableB_df = pd.read_csv(path.join(args.input, 'tableB.csv'), encoding_errors='replace')
    matches_df = pd.read_csv(path.join(args.input, 'matches.csv'), encoding_errors='replace')
    print("Table A: ", tableA_df.shape)
    print("Table B: ", tableB_df.shape)
    print("Matches: ", matches_df.shape)

    train, test = split_input(tableA_df, tableB_df, matches_df, recall=args.recall, seed=random.randint(0, 4294967295))
    print("Done!")
    print("Test set size: ", test.shape[0])
    print("Train set size: ", train.shape[0])

    train.to_csv(path.join(args.output, "train.csv"), index=False)
    test.to_csv(path.join(args.output, "test.csv"), index=False)
