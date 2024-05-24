import argparse
from os import path

import pathtype
import pandas as pd
from DeepBlocker.deep_blocker import DeepBlocker
from DeepBlocker.tuple_embedding_models import AutoEncoderTupleEmbedding
from DeepBlocker.vector_pairing_models import ExactTopKVectorPairing
from DeepBlocker.blocking_utils import compute_blocking_statistics


parser = argparse.ArgumentParser(description='Benchmark a dataset with a method')
parser.add_argument('input', type=pathtype.Path(readable=True), nargs='?', default='/data',
                    help='Input directory containing the dataset')
parser.add_argument('output', type=pathtype.Path(writable=True), nargs='?', default='/data/output',
                    help='Output directory to store the output')
parser.add_argument('-k', '--top-key', type=int, nargs='?', default=31,
                    help='Number of top vector keys to use')
parser.add_argument('-d', '--delimiter', type=str, default='|',
                    help='Delimiter to use in files')

args = parser.parse_args()

left_df = pd.read_csv(path.join(args.input, "tableA.csv"), sep=args.d)
left_df['Name'] = left_df['Name'].astype(str)

right_df = pd.read_csv(path.join(args.input, "tableB.csv"), sep=args.d)
right_df['Name'] = right_df['Name'].astype(str)

golden_df = pd.read_csv(path.join(args.input, "matches.csv"), sep=args.d)
cols_to_block = ["Name"]

tuple_embedding_model = AutoEncoderTupleEmbedding()
topK_vector_pairing_model = ExactTopKVectorPairing(K=args.k)
db = DeepBlocker(tuple_embedding_model, topK_vector_pairing_model)
candidate_set_df = db.block_datasets(left_df, right_df, cols_to_block)

results = compute_blocking_statistics(candidate_set_df, golden_df, left_df, right_df)
candidate_set_df.to_csv(path.join(args.output, "candidates.csv"), sep=args.d, index=False)

print(results)
