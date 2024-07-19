import argparse
#import pathtype
import random
import os

import py_entitymatching as em
from transform import transform_input, transform_output
from feature_extraction import gather_features_and_labels, gather_similarity_features
import numpy as np
import utils
import time

parser = argparse.ArgumentParser(description='Benchmark a dataset with a method')
parser.add_argument('input', nargs='?', default='datasets/d2_abt_buy', #, type=pathtype.Path(readable=True)
                    help='Input directory containing the dataset')
parser.add_argument('output',  nargs='?', default='output', #type=pathtype.Path(writable=True),
                    help='Output directory to store the output')
parser.add_argument('-T', '--transitivity', action='store_true',
                    help="whether to enforce transitivity constraint")
parser.add_argument('-f', '--full', action='store_true',
                    help='To perform matching on the full dataset or only on the test set')

args = parser.parse_args()

print("Hi, I'm ZeroER entrypoint!")
print("Input directory: ", os.listdir(args.input))
print("Output directory: ", os.listdir(args.output))

excl_attributes = ['_id', 'ltable_id', 'rtable_id', 'label']

def add_catalog_information(df, tableA, tableB):
    em.set_ltable(df, tableA)
    em.set_rtable(df, tableB)
    em.set_fk_ltable(df, excl_attributes[1])
    em.set_fk_rtable(df, excl_attributes[2])
    em.set_key(df, excl_attributes[0])
    
read_prefixes = ['tableA_', 'tableB_']

dataset, tableA, tableB, GT = transform_input(args.input, read_prefixes, args.full)

if args.full:
    exp_data = dataset
else:
    # if use_full_dataset = False, only uses test part of the dataset
    exp_data = dataset[1]

em.set_key(tableA, 'id')
em.set_key(tableB, 'id')
add_catalog_information(exp_data, tableA, tableB)

id_df = exp_data[["ltable_id", "rtable_id"]]
cand_features = gather_features_and_labels(tableA, tableB, GT, exp_data)
sim_features = gather_similarity_features(cand_features)
sim_features_lr = (None,None)
id_dfs = (None, None, None)
if args.transitivity == True:
    id_dfs = (id_df, None, None)


true_labels = cand_features.gold.values
if np.sum(true_labels)==0:
    true_labels = None

start_time = time.perf_counter()
y_pred = utils.run_zeroer(sim_features, sim_features_lr,id_dfs,
                    true_labels ,True,False,args.transitivity)
end_time = time.perf_counter()

pred_df = cand_features.copy()
pred_df['pred'] = y_pred

transform_output(pred_df, end_time-start_time, args.output)
