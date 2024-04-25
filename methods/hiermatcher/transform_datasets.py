import csv
import os

# Source to destination mapping
dataset_dir = [
  ('../existing_datasets/Structured/DBLP-ACM/', 'Ds1/'),
  ('../existing_datasets/Structured/DBLP-GoogleScholar/', 'Ds2/'),
  ('../existing_datasets/Structured/iTunes-Amazon/', 'Ds3/'),
  ('../existing_datasets/Structured/Walmart-Amazon/', 'Ds4/'),
  ('../existing_datasets/Structured/Beer/', 'Ds5/'),
  ('../existing_datasets/Structured/Amazon-Google/', 'Ds6/'),
  ('../existing_datasets/Structured/Fodors-Zagats/', 'Ds7/'),
  ('../existing_datasets/Dirty/DBLP-ACM/', 'Dd1/'),
  ('../existing_datasets/Dirty/DBLP-GoogleScholar/', 'Dd2/'),
  ('../existing_datasets/Dirty/iTunes-Amazon/', 'Dd3/'),
  ('../existing_datasets/Dirty/Walmart-Amazon/', 'Dd4/'),
  ('../existing_datasets/Textual/Abt-Buy/', 'Dt1/'),
  ('../existing_datasets/Textual/Company/', 'Dt2/')
]

# Iterate data sets
for source_dir, dest_dir in dataset_dir:

  if not os.path.isdir(dest_dir):
    os.mkdir(dest_dir)

  tableA = csv.DictReader(open(source_dir + 'tableA.csv'))
  tableB = csv.DictReader(open(source_dir + 'tableB.csv'))
  tableA_dict = {row['id']: row for row in tableA}
  tableB_dict = {row['id']: row for row in tableB}

  attributes = tableA.fieldnames
  attributes.remove('id')

  for file_name in ['test.csv', 'train.csv', 'val.csv']:

    dataset_rows = csv.DictReader(open(source_dir + file_name, 'r'))
    dest_attributes = ['id', 'label']

    for a in attributes:
      dest_attributes.append('left_' + a)
      dest_attributes.append('right_' + a)

    id = 0
    rows = list()
    for record_pair in dataset_rows:
      l_record = tableA_dict[record_pair['ltable_id']]
      r_record = tableB_dict[record_pair['rtable_id']]

      dest_row = {
        'id': id,
        'label': record_pair['label'],
      }
      for a in attributes:
        dest_row['left_' + a] = l_record[a]
        dest_row['right_' + a] = r_record[a]
      rows.append(dest_row)
      id += 1

    w = csv.DictWriter(open(dest_dir + file_name, 'a'), dest_attributes)
    w.writeheader()
    w.writerows(rows)
