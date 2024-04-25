import sys
import csv
import os
import shutil


def transform_dataset(source_dir, dest_dir):
    # Copy tableA.csv and tableB.csv
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


if __name__ == "__main__":
    in_path = sys.argv[1]
    out_path = sys.argv[2]

    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    transform_dataset(in_path, out_path)
