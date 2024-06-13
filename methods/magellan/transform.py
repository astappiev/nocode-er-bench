import os
import pandas as pd


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
