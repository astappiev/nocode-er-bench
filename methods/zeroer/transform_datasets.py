import pandas as pd
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

# Iterate data sets
for source_dir, dest_dir in dataset_dir:

  if not os.path.isdir(dest_dir):
    os.mkdir(dest_dir)

  # Create a metadata file
  with open(dest_dir + 'metadata.txt', 'w') as file:
    file.write('tableA.csv\n')
    file.write('tableB.csv\n')
    file.write('matches_tableA_tableB.csv')


  # Copy tableA.csv, tableB.csv, and train.csv
  shutil.copyfile(source_dir + 'tableA.csv', dest_dir + '/tableA.csv')
  shutil.copyfile(source_dir + 'tableB.csv', dest_dir + '/tableB.csv')
  shutil.copyfile(source_dir + 'train.csv', dest_dir + '/train.csv')

  # Create the matches file
  df = pd.read_csv(open(source_dir + 'train.csv', 'r'))
  df_matches = df[df['label'] == 1]
  df_matches = df_matches[['ltable_id', 'rtable_id']]
  df_matches.to_csv(open(dest_dir + 'matches_tableA_tableB.csv', 'w'))
