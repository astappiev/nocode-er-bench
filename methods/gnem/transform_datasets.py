import os
import shutil

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

for source_dir, dest_dir in dataset_dir:

  if not os.path.isdir(dest_dir):
    os.mkdir(dest_dir)

  for file_name in ['test.csv', 'train.csv', 'val.csv']:
    source_file = source_dir + file_name
    dest_file = dest_dir + file_name
    shutil.copy2(source_file, dest_file)
