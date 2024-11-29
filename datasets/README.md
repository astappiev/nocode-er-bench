# Datasets

## Structure

All the datasets should consist of 3 files: `tableA.csv`, `tableB.csv`, and `matches.csv`.

- They are normalised comma-separated files, where the first row is the header;
- `tableA.csv` and `tableB.csv` should have the same attributes, `id` is the only a required attribute;
- `matches.csv` should have the following attributes: `tableA_id`, `tableB_id`;
- Each row in `matches.csv` should contain the ids of the records that are matches in the tables.

## Splitter

In the first iteration, we process the dataset with a splitter, which generates the `train.csv` and `test.csv` datasets.
Then the methods receive all 5 files: `tableA.csv`, `tableB.csv`, `matches.csv`, `train.csv`, and `test.csv` as input.

## Sources
- https://github.com/AI-team-UoA/pyJedAI/tree/main/data/ccer
- https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md
- https://zenodo.org/records/8164151/files/magellanExistingDatasets.tar.gz
