import csv
import os
import shutil
import sys


def transform_input(source_dir, dest_dir):
    # The source directory contains the following files (in common format for all methods):
    # - tableA.csv (where the first row is the header, and it has to contain the id attribute)
    # - tableB.csv (same as tableA.csv)
    # - matches.csv (should have tableA_id, tableB_id attributes, which means that the tableA_id record is a match with the tableB_id record)
    #
    # The output directory should contain the files converted into the format expected by the method.

    shutil.copyfile(os.path.join(source_dir, 'tableA.csv'), os.path.join(dest_dir, 'tableA.csv'))
    shutil.copyfile(os.path.join(source_dir, 'tableB.csv'), os.path.join(dest_dir, 'tableB.csv'))

    tableA = csv.DictReader(open(os.path.join(source_dir, 'tableA.csv'), 'r'))
    tableB = csv.DictReader(open(os.path.join(source_dir, 'tableB.csv'), 'r'))
    tableA_dict = {row['id']: row for row in tableA}
    tableB_dict = {row['id']: row for row in tableB}

    attributes = tableA.fieldnames

    for file_name in ['test.csv', 'train.csv', 'val.csv']:
        dataset_rows = csv.DictReader(open(os.path.join(source_dir, file_name), 'r'))

        dest_attributes = ['_id', 'label']
        for a in attributes:
            dest_attributes.append('table1.' + a)
            dest_attributes.append('table2.' + a)

        id = 0
        rows = list()
        for record_pair in dataset_rows:
            l_record = tableA_dict[record_pair['ltable_id']]
            r_record = tableB_dict[record_pair['rtable_id']]

            tr_row = {
                '_id': id,
                'label': record_pair['label'],
            }
            for a in attributes:
                tr_row['table1.' + a] = l_record[a]
                tr_row['table2.' + a] = r_record[a]
            rows.append(tr_row)
            id += 1

        w = csv.DictWriter(open(os.path.join(dest_dir, file_name), 'w'), dest_attributes)
        w.writeheader()
        w.writerows(rows)


def transform_output(source_dir, dest_dir):
    # The source directory contains the output from the method.
    # This should be converted to one common format from all methods and stored into the destination directory.
    # metrics.csv: F1, Precision, Recall, Time (1 row, 4 columns)
    # predictions.csv: tableA_id, tableB_id, etc. (should have at least 2 columns and a header row)

    # TODO: Implement this function
    return None


if __name__ == "__main__":
    in_path = sys.argv[1]
    out_path = sys.argv[2]

    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    transform_input(in_path, out_path)
