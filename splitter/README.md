# DeepBlocker dataset splitter

Splits the dataset into training, validation, and testing sets.
Based on [DeepBlocker](https://github.com/gpapadis/DLMatchers/tree/main/DeepBlocker4NewDatasets).

## Expected directory structure
It is expected that it contains the following files (in proper CSV format):
 - `tableA.csv` where the first row is the header, and it has to contain the `id` attribute
 - `tableB.csv` same as `tableA.csv`
 - `matches.csv` should have `tableA_id`, `tableB_id` attributes, which means that the `tableA_id` record is a match with the `tableB_id` record

The produced output will include two files:
- `test.csv` where attributes are: `tableA_id`, `tableB_id` and `label` (0 or 1). The label is 1 if the pair is a match, 0 otherwise.
- `train.csv` same as `test.csv`, and the split is 70% train and 30% test.

## How to use

You can directly execute the docker image as following:
```bash
docker run --rm -v .:/data splitter
```
This will assume that you have the input dataset in the current directory,
it will mount it as `/data` and will output the results in the same folder.

You can override the input and output directories by providing them as arguments to the docker image:
```bash
docker run -v ../../datasets/d2_abt_buy:/data/input:ro -v ../../test:/data/output splitter /data/input /data/output
```
