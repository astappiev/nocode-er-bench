import os
import sys

import pandas as pd


def join_columns(table, columns_to_join=None, separator=' ', prefixes=['tableA_', 'tableB_']):
    agg_table = pd.DataFrame()
    for prefix in prefixes:
        print(columns_to_join)
        if columns_to_join == None:
            columns = [column for column in table.columns if (column != prefix + 'id' and prefix in column)]
        else:
            columns = [prefix + column for column in columns_to_join]
        print(columns)

        red_table = table.loc[:, columns]
        red_table = red_table.fillna('')
        red_table = red_table.astype(str)

        part_table = red_table.aggregate(separator.join, axis=1)
        # part_table = part_table.map(lambda x: x.replace('nan', ''))
        part_table.rename(prefix + 'AgValue', inplace=True)

        agg_table = pd.concat([agg_table, table[prefix + 'id'], part_table], axis=1)

    return pd.concat([agg_table, table['label']], axis=1)


def transform_input(source_dir, columns_to_join=None, separator=' ', prefixes=['tableA_', 'tableB_']):
    train_df = pd.read_csv(os.path.join(source_dir, 'train.csv'), encoding_errors='replace')
    test_df = pd.read_csv(os.path.join(source_dir, 'test.csv'), encoding_errors='replace')

    train = join_columns(train_df, columns_to_join, separator, prefixes)
    test = join_columns(test_df, columns_to_join, separator, prefixes)

    return train, test


def transform_output(predictions_df, test_table, runtime, dest_dir):
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
    candidate_ids = predictions_df[predictions_df['predictions'] == 1]
    candidate_table = test_table.iloc[candidate_ids.index]
    # save candidate pair IDs to predictions.csv
    candidate_table[['tableA_id', 'tableB_id']].to_csv(os.path.join(dest_dir, 'predictions.csv'), index=False)
    
    if candidate_table.shape[0] > 0:
        # calculate evaluation metrics
        num_candidates = candidate_table.shape[0]
        true_positives = candidate_table['label'].sum()
        print(candidate_table[candidate_table['label'] == 1])
    
        ground_truth = test_table['label'].sum()
    
        recall = true_positives / ground_truth
        precision = true_positives / num_candidates
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
        precision = 0
        recall = 0

    # save evaluation metrics to metrics.csv
    pd.DataFrame({
        'f1': [f1],
        'precision': [precision],
        'recall': [recall],
        'time': [runtime],
    }).to_csv(os.path.join(dest_dir, 'metrics.csv'), index=False)
    return None


if __name__ == "__main__":
    in_path = sys.argv[1]
    out_path = sys.argv[2]

    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    transform_input(in_path)
