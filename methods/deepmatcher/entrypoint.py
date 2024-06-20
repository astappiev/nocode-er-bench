import argparse
import resource
import time
import os
import pathtype

import pandas as pd
import deepmatcher as dm
from transform import transform_output

parser = argparse.ArgumentParser(description='Benchmark a dataset with a method')
parser.add_argument('input', type=pathtype.Path(readable=True), nargs='?', default='/data',
                    help='Input directory containing the dataset')
parser.add_argument('output', type=pathtype.Path(writable=True), nargs='?', default='/data/output',
                    help='Output directory to store the output')
parser.add_argument('embedding', type=pathtype.Path(readable=True), nargs='?', default='/workspace/embedding',
                    help='The directory where embeddings are stored')

args = parser.parse_args()

print("Hi, I'm DeepMatcher entrypoint!")
print("Input directory: ", os.listdir(args.input))
print("Output directory: ", os.listdir(args.output))

train = pd.read_csv(os.path.join(args.input, 'train.csv'), encoding_errors='replace')
test = pd.read_csv(os.path.join(args.input, 'test.csv'), encoding_errors='replace')

train.drop(columns=['tableA_id', 'tableB_id']).to_csv(os.path.join(args.output, 'train.csv'), index_label='id')
test.drop(columns=['tableA_id', 'tableB_id']).to_csv(os.path.join(args.output, 'test.csv'), index_label='id')

# Step 1. Convert input data into the format expected by the method
datasets = dm.data.process(path=args.output,
                           train="train.csv",
                           test="test.csv",
                           id_attr='id',
                           label_attr='label',
                           left_prefix='tableA_',
                           right_prefix='tableB_',
                           #cache=None,
                           embeddings_cache_path=args.embedding)

train, test = datasets[0], datasets[1] if len(datasets) >= 2 else None

# Step 2. Run the method
model = dm.MatchingModel()

start_time = time.time()
model.run_train(train, test, epochs=15, best_save_path='best_model.pth')
train_time = time.time() - start_time
train_max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

t_start = time.time()
stats = model.run_eval(test, return_stats=True)
test_time = time.time() - start_time
test_max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

# Step 3. Convert the output into a common format
transform_output(stats, test_time, test_max_mem, args.output)
print("Final output: ", os.listdir(args.output))
