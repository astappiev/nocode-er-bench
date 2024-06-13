import os
import pandas as pd


def transform_output(stats, runtime, max_mem, dest_dir):
    # save evaluation metrics to metrics.csv
    pd.DataFrame({
        'f1': [stats.f1()],
        'precision': [stats.precision()],
        'recall': [stats.recall()],
        'max_mem': [max_mem],
        'time': [runtime],
    }).to_csv(os.path.join(dest_dir, 'metrics.csv'), index=False)
    return None
