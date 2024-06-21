#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:59:51 2024

@author: franziska
"""

import os
import argparse
import random
import json
import sys
import torch
import numpy as np
from collections import namedtuple
import time
from tqdm import tqdm


from transform import transform_input_old, transform_output

sys.path.insert(0, "Snippext_public")

from dataset import DittoDataset
from summarize import Summarizer
from knowledge import ProductDKInjector, GeneralDKInjector
from ditto import train
from matcher import classify, tune_threshold

from scipy.special import softmax


parser = argparse.ArgumentParser(description='Benchmark a dataset with a method')
parser.add_argument('input', nargs='?', default='datasets/d2_abt_buy',
                    help='Input directory containing the dataset')
parser.add_argument('temp_output', nargs='?', default='output/temp',
                    help='directory to save temporary files')
parser.add_argument('output', nargs='?', default='output',
                    help='Output directory to store the output')
parser.add_argument('-r', '--recall', type=float, nargs='?', default=0.8,
                    help='Recall value used to select ground truth pairs')
parser.add_argument('-s', '--seed', type=int, nargs='?', default=random.randint(0, 4294967295),
                    help='The random state used to initialize the algorithms and split dataset')
parser.add_argument("--run_id", type=int, default=0)

parser.add_argument("--model", type=str, default='distilbert')
parser.add_argument('--epochs', default=1, type=float)

args = parser.parse_args()

print("Hi, I'm DITTO entrypoint!")
print("Input directory: ", os.listdir(args.input))
print("Output directory: ", os.listdir(args.output))

# Step 1. Convert input data into the format expected by the method
print("Method input: ", os.listdir(args.input))
prefix_1 = 'tableA_'
prefix_2 = 'tableB_'
trainset, testset, train_ids, test_ids = transform_input_old(args.input, args.temp_output, args.recall, seed=args.seed)

hyperparameters = namedtuple('hyperparameters', ['lm', #language Model
                                                 'n_epochs', #number of epochs
                                                 'batch_size',
                                                 'max_len', #max number of tokens as input for language model
                                                 'lr', #learning rate 
                                                 'save_model',
                                                 'logdir',
                                                 'fp16', #train with half precision
                                                 'da', #data augmentation
                                                 'alpha_aug',
                                                 'dk', #domain knowledge
                                                 'summarize', #summarize to max_len
                                                 'size',#dataset size
                                                 'run_id']) 

hp = hyperparameters(lm = args.model,
                     n_epochs = args.epochs,
                     batch_size = 64,
                     max_len = 256,
                     lr = 3e-5,
                     save_model = False,
                     logdir = args.temp_output,
                     fp16 = False,
                     da = 'all',
                     alpha_aug = 0.8,
                     dk = 'general',
                     summarize = True,
                     size = None,
                     run_id = args.run_id)
                     



#parser.add_argument("--finetuning", dest="finetuning", action="store_true")



# set seeds
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# only a single task for baseline
task = args.input

# create the tag of the run
run_tag = '%s_lm=%s_da=%s_dk=%s_su=%s_size=%s_id=%d' % (task, args.model, hp.da,
        hp.dk, hp.summarize, str(hp.size), hp.run_id)
run_tag = run_tag.replace('/', '_')

# # load task configuration
# configs = json.load(open('configs.json'))
# configs = {conf['name'] : conf for conf in configs}
# config = configs[task]

# trainset = config['trainset']
# #validset = config['validset']
# testset = config['testset']

# summarize the sequences up to the max sequence length
if hp.summarize:
    summarizer = Summarizer([trainset], lm=args.model)
    trainset = summarizer.transform_file(trainset, max_len=hp.max_len)
    #validset = summarizer.transform_file(validset, max_len=args.max_len)
    testset = summarizer.transform_file(testset, max_len=hp.max_len)

if hp.dk is not None:
    if hp.dk == 'product':
        injector = ProductDKInjector(hp.dk)
    else:
        injector = GeneralDKInjector(hp.dk)

    trainset = injector.transform_file(trainset)
    #validset = injector.transform_file(validset)
    testset = injector.transform_file(testset)

# load train/dev/test sets
train_dataset = DittoDataset(trainset,
                               lm=args.model,
                               max_len=hp.max_len,
                               size=hp.size,
                               da=hp.da)
#valid_dataset = DittoDataset(validset, lm=args.model)
test_dataset = DittoDataset(testset, lm=args.model)

# train and evaluate the model
matcher = train(train_dataset,
      run_tag, hp)

pairs = []

threshold = 0.5




# batch processing
out_data = []
start_time = time.time()
predictions, logits = classify(test_dataset, matcher, lm=hp.lm,
                               batch_size = hp.batch_size,
                               max_len=hp.max_len,
                               threshold=threshold)
scores = softmax(logits, axis=1)

runtime = 0
transform_output(predictions, test_ids, test_dataset.labels, runtime, args.output)



run_time = time.time() - start_time
run_tag = '%s_lm=%s_dk=%s_su=%s' % (task, hp.lm, str(hp.dk != None), str(hp.summarize != None))
os.system('echo %s %f >> log.txt' % (run_tag, run_time))
