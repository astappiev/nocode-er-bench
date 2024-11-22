import argparse
from os import path
import random
import pathtype
import pandas as pd
from deep_blocker import DeepBlocker
from tuple_embedding_models import AutoEncoderTupleEmbedding
from vector_pairing_models import ExactTopKVectorPairing
from sklearn.model_selection import train_test_split


def generate_candidates(embedding_path, tableA_df, tableB_df, matches_df, top_key=31):
    cols_to_block = list(set(tableA_df.columns.tolist()) & set(tableB_df.columns.tolist()))
    cols_to_block.remove('id')
    print("Blocking columns: ", cols_to_block)

    golden_df = matches_df.rename(columns={'tableA_id': 'ltable_id', 'tableB_id': 'rtable_id'})

    tuple_embedding_model = AutoEncoderTupleEmbedding(embedding_path=embedding_path)
    topK_vector_pairing_model = ExactTopKVectorPairing(K=top_key)
    db = DeepBlocker(tuple_embedding_model, topK_vector_pairing_model)
    candidate_set_df = db.block_datasets(tableA_df, tableB_df, cols_to_block)

    golden_df['label'] = 1
    candidate_set_df['label'] = 0
    pairs_df = pd.concat([golden_df, candidate_set_df]).drop_duplicates(subset=['ltable_id', 'rtable_id'], keep='first')

    cand_tableA = tableA_df.add_prefix('tableA_')
    cand_tableB = tableB_df.add_prefix('tableB_')

    return pd.concat([
        (cand_tableA.iloc[pairs_df['ltable_id']]).reset_index(drop=True),
        (cand_tableB.iloc[pairs_df['rtable_id']]).reset_index(drop=True),
        pairs_df['label'].reset_index(drop=True)
    ], axis=1)


def split_input(embedding_path, tableA_df, tableB_df, matches_df, recall=0.7, top_key=31, seed=1):
    candidates = generate_candidates(embedding_path, tableA_df, tableB_df, matches_df, top_key=top_key)
    print("Candidates generated: ", candidates.shape[0])
    return train_test_split(candidates, train_size=recall, random_state=seed, shuffle=True, stratify=candidates['label'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Splits the dataset using DeepBlocker method')
    parser.add_argument('input', type=pathtype.Path(readable=True), nargs='?', default='/data',
                        help='Input directory containing the dataset')
    parser.add_argument('output', type=pathtype.Path(writable=True), nargs='?',
                        help='Output directory to store the output. If not provided, the input directory will be used')
    parser.add_argument('embedding', type=pathtype.Path(readable=True), nargs='?', default='/workspace/embedding',
                    help='The directory where embeddings are stored')
    parser.add_argument('-r', '--recall', type=float, nargs='?', default=0.7,
                        help='The recall value for the train set')
    parser.add_argument('-k', '--top-key', type=int, nargs='?', default=31,
                        help='Number of top vector keys to use')
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input

    print("Hi, I'm DeepBlocker splitter, I'm doing random split of the input datasets into train and test sets.")
    tableA_df = pd.read_csv(path.join(args.input, 'tableA.csv'), encoding_errors='replace')
    tableB_df = pd.read_csv(path.join(args.input, 'tableB.csv'), encoding_errors='replace')
    matches_df = pd.read_csv(path.join(args.input, 'matches.csv'), encoding_errors='replace')
    print("Input tables are:", "A", tableA_df.shape, "B", tableB_df.shape, "Matches", matches_df.shape)

    train, test = split_input(str(args.embedding), tableA_df, tableB_df, matches_df, recall=args.recall, top_key=args.top_key, seed=random.randint(0, 4294967295))
    print("Done! Train size: {}, test size: {}.".format(train.shape[0], test.shape[0]))

    train.to_csv(path.join(args.output, "train.csv"), index=False)
    test.to_csv(path.join(args.output, "test.csv"), index=False)
