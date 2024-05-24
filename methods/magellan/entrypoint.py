import os
import argparse
import logging
import time
import resource
import csv

import fcntl
import pathtype
import numpy as np
import py_entitymatching as em

from transform import transform_input, transform_output


parser = argparse.ArgumentParser(description='Benchmark a dataset with a method')
parser.add_argument('input', type=pathtype.Path(readable=True), nargs='?', default='/data',
                    help='Input directory containing the dataset')
parser.add_argument('output', type=pathtype.Path(writable=True), nargs='?', default='/data/output',
                    help='Output directory to store the output')
parser.add_argument('-t', '--temp', type=pathtype.Path(writable_or_creatable=True), nargs='?', default='/tmpdir',
                    help='A folder to store temporary files')
parser.add_argument('-r', '--recall', type=int, nargs='?',
                    help='Recall value for the algorithm')
parser.add_argument('-e', '--epochs', type=int, nargs='?',
                    help='Number of epochs for the algorithm')

args = parser.parse_args()

print("Hi, I'm Magellan entrypoint!")
print("Input directory: ", os.listdir(args.input))
print("Output directory: ", os.listdir(args.output))

# Create a temporary directory to store intermediate files
# FIXME: Could be removed if not needed (e.g. if the method does not require any intermediate files or data stored in memory)
temp_input = os.path.join(args.temp, 'input')
temp_output = os.path.join(args.temp, 'output')

# Step 1. Convert input data into the format expected by the method
# FIXME: feel free to change the number of arguments, or store the data in variables and return them from the function
transform_input(args.input, temp_input)
print("Method input: ", os.listdir(temp_input))

# Step 2. Run the method

def extract_feature_vectors(tableA, tableB, pairs, train_feature_subset=None):
    start_time = time.time()

    logging.info('Number of tuples in A: ' + str(len(tableA)))
    logging.info('Number of tuples in B: ' + str(len(tableB)))
    logging.info('Number of tuples in A x B (i.e the cartesian product): ' + str(len(tableA) * len(tableB)))

    if train_feature_subset is None:
        feature_table = em.get_features_for_matching(tableA, tableB, validate_inferred_attr_types=False)

        # Remove ID based features
        logging.info('All features {}'.format(feature_table['feature_name']))
        train_feature_subset = feature_table[4:]
        logging.info('Selected features {}'.format(train_feature_subset))

    # Generate features
    feature_vectors = em.extract_feature_vecs(pairs, feature_table=train_feature_subset, attrs_after='label')

    # Impute feature vectors with the mean of the column values.
    feature_vectors = em.impute_table(feature_vectors,
                                      exclude_attrs=['_id', 'table1.id', 'table2.id', 'label'],
                                      strategy='mean', missing_val=np.NaN)
    t = time.time() - start_time
    return feature_vectors, t, train_feature_subset


def predict(train, test, time_train, time_test, result_file):
    """
    Predict results using four different classifiers of Magellan
    and average the results

    :param train: Train data frame
    :param test: Test data frame
    :return:
    """
    # Create a set of ML-matchers
    dt = em.DTMatcher(name='DecisionTree')
    svm = em.SVMMatcher(name='SVM')
    rf = em.RFMatcher(name='RF')
    lg = em.LogRegMatcher(name='LogReg')

    # Train and eval on different classifiers
    for clf, clf_name in [(dt, 'DecisionTree'), (svm, 'SVM'), (rf, 'RF'), (lg, 'LogReg')]:
        start_time = time.time()
        clf.fit(table=train,
                exclude_attrs=['_id', 'table1.id', 'table2.id', 'label'],
                target_attr='label')
        time_train += time.time() - start_time
        train_max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        # Predict M
        start_time = time.time()
        predictions = clf.predict(table=test,
                                  exclude_attrs=['_id', 'table1.id', 'table2.id', 'label'],
                                  append=True, target_attr='predicted',
                                  inplace=False)

        # Evaluate the result
        eval_result = em.eval_matches(predictions, 'label', 'predicted')
        time_test += time.time() - start_time
        test_max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        em.print_eval_summary(eval_result)

        p = eval_result['precision']
        r = eval_result['recall']
        if (p + r - p * r) == 0:
            f_star = 0
        else:
            f_star = p * r / (p + r - p * r)
        logging.info('---{} p{} r{} fst{}'.format(clf_name, p, r, f_star))

        p = round(p * 100, 2)
        r = round(r * 100, 2)
        f_star = round(f_star * 100, 2)

        file_exists = os.path.isfile(result_file)
        with open(result_file, 'a') as results_file:
            heading_list = ['method', 'train_time', 'test_time',
                            'train_max_mem', 'test_max_mem', 'TP', 'FP', 'FN',
                            'TN', 'Pre', 'Re', 'F1', 'Fstar']
            writer = csv.DictWriter(results_file, fieldnames=heading_list)

            if not file_exists:
                writer.writeheader()

            fcntl.flock(results_file, fcntl.LOCK_EX)
            result_dict = {
                'method': 'magellan' + clf_name,
                'train_time': round(time_train, 2),
                'test_time': round(time_test, 2),
                'train_max_mem': train_max_mem,
                'test_max_mem': test_max_mem,
                'TP': eval_result['pred_pos_num'] - eval_result['false_pos_num'],
                'FP': eval_result['false_pos_num'],
                'FN': eval_result['false_neg_num'],
                'TN': eval_result['pred_neg_num'] - eval_result['false_neg_num'],
                'Pre': ('{prec:.2f}').format(prec=p),
                'Re': ('{rec:.2f}').format(rec=r),
                'F1': ('{f1:.2f}').format(f1=round(eval_result['f1'] * 100, 2)),
                'Fstar': ('{fstar:.2f}').format(fstar=f_star)
            }
            writer.writerow(result_dict)
            fcntl.flock(results_file, fcntl.LOCK_UN)


tableA = em.read_csv_metadata(os.path.join(args.input, 'tableA.csv'), key='id')
tableB = em.read_csv_metadata(os.path.join(args.input, 'tableB.csv'), key='id')

trainPairs = em.read_csv_metadata(os.path.join(args.input, 'train.csv'), key='_id', ltable=tableA, rtable=tableB, fk_ltable='tableA_id', fk_rtable='tableB_id')
testPairs = em.read_csv_metadata(os.path.join(args.input, 'test.csv'), key='_id', ltable=tableA, rtable=tableB, fk_ltable='tableA_id', fk_rtable='tableB_id')

# Generate training feature vectors using full data set
train, t_train, train_feature_subset = extract_feature_vectors(tableA, tableB, trainPairs)

# Generate testing feature vectors using only the specified link
test, t_test, _ = extract_feature_vectors(tableA, tableB, testPairs, train_feature_subset)

result_file = os.path.join(temp_output, 'results.csv')

predict(train, test, t_train, t_test, result_file)
print("Method output: ", os.listdir(temp_output))

# Step 3. Convert the output into a common format
# FIXME: feel free to change the number of arguments, but make sure to persist the data to the output directory
transform_output(temp_output, args.output)
print("Final output: ", os.listdir(args.output))
