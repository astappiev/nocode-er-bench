#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 11:09:50 2024

@author: franziska
"""

import os
import sys

import pandas as pd
import numpy as np
import utils



def transform_input(source_dir, prefixes=['tableA_', 'tableB_'], use_full=True):
    
    tableA_df = pd.read_csv(os.path.join(source_dir, 'tableA.csv'))
    tableB_df = pd.read_csv(os.path.join(source_dir, 'tableB.csv'))
    matches_df = pd.read_csv(os.path.join(source_dir, 'matches.csv'))
    
    train_df = pd.read_csv(os.path.join(source_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(source_dir, 'test.csv'))
    
    if use_full:
        pair_df = pd.concat([train_df, test_df])
        pair_df.rename(lambda x: x.replace(prefixes[0], 'ltable_'), axis='columns', inplace=True)
        pair_df.rename(lambda x: x.replace(prefixes[1], 'rtable_'), axis='columns', inplace=True)
        
        return pair_df, tableA_df, tableB_df, matches_df
    
    
    train_df.rename(lambda x: x.replace('tableA', 'ltable'), axis='columns', inplace=True)
    train_df.rename(lambda x: x.replace('tableB', 'rtable'), axis='columns', inplace=True)
    
    test_df.rename(lambda x: x.replace('tableA', 'ltable'), axis='columns', inplace=True)
    test_df.rename(lambda x: x.replace('tableB', 'rtable'), axis='columns', inplace=True)
    
    matches_df.rename(lambda x: x.replace('tableA', 'ltable'), axis='columns', inplace=True)
    matches_df.rename(lambda x: x.replace('tableB', 'rtable'), axis='columns', inplace=True)
    
    return (train_df, test_df), tableA_df, tableB_df, matches_df



def transform_output(predictions, runtime, dest_dir):
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

    predictions['prediction'] = np.round(np.clip(predictions['pred'] + utils.DEL, 0., 1.)).astype(int)
    
    
    candidate_table = predictions[predictions['prediction'] == 1]
    print(candidate_table)
    # save candidate pair IDs to predictions.csv
    p_table = candidate_table[['ltable_id', 'rtable_id']]
    p_table.columns = ['tableA_id', 'tableB_id']
    p_table.to_csv(os.path.join(dest_dir, 'predictions.csv'), index=False)

    # calculate evaluation metrics
    num_candidates = candidate_table.shape[0]
    true_positives = candidate_table['gold'].sum()
    ground_truth = predictions['gold'].sum()

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