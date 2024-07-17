# EMTransformer

https://github.com/gpapadis/DLMatchers/tree/main/EMTransformer

## How to use

You can directly execute the docker image as following:

```bash
docker run --rm -v .:/data emtransformer
```

This will assume that you have the input dataset in the current directory,
it will mount it as `/data` and will output the results in the `output` subdirectory.

You can override the input and output directories by providing them as arguments to the docker image:

```bash
docker run -v ../../datasets/d2_abt_buy:/data/input:ro -v ../../test:/data/output emtransformer /data/input /data/output
```

Has Error:
```
Hi, I'm EMTransformer entrypoint!
2024-07-17T12:51:13.052118761Z Input directory:  ['matches.csv', 'tableA.csv', 'tableB.csv', 'test.csv', 'train.csv']
2024-07-17T12:51:13.053371694Z Output directory:  ['cacheddata.pth', 'test.csv', 'train.csv']
2024-07-17T12:51:13.058138785Z Method input:  ['matches.csv', 'tableA.csv', 'tableB.csv', 'test.csv', 'train.csv']
2024-07-17T12:51:13.175610637Z Traceback (most recent call last):
2024-07-17T12:51:13.175682529Z   File "/opt/conda/lib/python3.7/site-packages/pandas/core/indexes/base.py", line 3361, in get_loc
2024-07-17T12:51:13.176538938Z     return self._engine.get_loc(casted_key)
2024-07-17T12:51:13.176573746Z   File "pandas/_libs/index.pyx", line 76, in pandas._libs.index.IndexEngine.get_loc
2024-07-17T12:51:13.176885926Z   File "pandas/_libs/index.pyx", line 108, in pandas._libs.index.IndexEngine.get_loc
2024-07-17T12:51:13.176920524Z   File "pandas/_libs/hashtable_class_helper.pxi", line 5198, in pandas._libs.hashtable.PyObjectHashTable.get_item
2024-07-17T12:51:13.177060690Z   File "pandas/_libs/hashtable_class_helper.pxi", line 5206, in pandas._libs.hashtable.PyObjectHashTable.get_item
2024-07-17T12:51:13.177092874Z KeyError: 'id'
2024-07-17T12:51:13.177100709Z 
2024-07-17T12:51:13.177106441Z The above exception was the direct cause of the following exception:
2024-07-17T12:51:13.177112863Z 
2024-07-17T12:51:13.177118254Z Traceback (most recent call last):
2024-07-17T12:51:13.177123935Z   File "./entrypoint.py", line 41, in <module>
2024-07-17T12:51:13.177318428Z     train_df, test_df = transform_input(args.input, columns_to_join, ' ', [prefix_1, prefix_2])
2024-07-17T12:51:13.177333538Z   File "/workspace/transform.py", line 34, in transform_input
2024-07-17T12:51:13.177340652Z     train = join_columns(train_df, columns_to_join, separator, prefixes)
2024-07-17T12:51:13.177346454Z   File "/workspace/transform.py", line 8, in join_columns
2024-07-17T12:51:13.177492973Z     agg_table = table['id']
2024-07-17T12:51:13.177504075Z   File "/opt/conda/lib/python3.7/site-packages/pandas/core/frame.py", line 3458, in __getitem__
2024-07-17T12:51:13.178376728Z     indexer = self.columns.get_loc(key)
2024-07-17T12:51:13.178405294Z   File "/opt/conda/lib/python3.7/site-packages/pandas/core/indexes/base.py", line 3363, in get_loc
2024-07-17T12:51:13.179940585Z     raise KeyError(key) from err
2024-07-17T12:51:13.180101232Z KeyError: 'id'
```
