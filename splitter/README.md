# DeepBlocker dataset splitter

Splits the dataset into training, validation, and testing sets.
Based on [DeepBlocker](https://github.com/gpapadis/DLMatchers/tree/main/DeepBlocker4NewDatasets).

## Expected directory structure
It is expected that it contains the following files (in proper CSV format):
 - `tableA.csv` where the first row is the header, and it has to contain the `id` attribute
 - `tableB.csv` same as `tableA.csv`
 - `matches.csv` should have `tableA_id`, `tableB_id` attributes, which means that the `tableA_id` record is a match with the `tableB_id` record

The produced output will include two files, and the split by recall value provided (0.7 by default):
- `test.csv` where attributes are: `tableA_id`, `tableB_id` and `label` (0 or 1). The label is 1 if the pair is a match, 0 otherwise.
- `train.csv` same as `test.csv`

## How to use

IMPORTANT! `/workspace/embedding` should be mounted with `wiki.en.bin` embeddings inside.

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

## Apptainer

```bash
apptainer build ../apptainer/splitter.sif container.def
apptainer run ~/nocode-er-bench/apptainer/splitter.sif ~/nocode-er-bench/datasets/d1_fodors_zagats/ ~/nocode-er-bench/output/ ~/nocode-er-bench/embedding/
```

## Cuda error

```bash
❯ time apptainer run ../apptainer/splitter.sif ../datasets/d1_fodors_zagats/ ../output/ ../embedding/
Hi, I'm DeepBlocker splitter, I'm doing random split of the input datasets into train and test sets.
Input tables are: A (533, 7) B (331, 7) Matches (110, 2)
Blocking columns:  ['name', 'city', 'class', 'addr', 'type', 'phone']
Loading FastText model
Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
Performing pre-processing for tuple embeddings 
Training AutoEncoder model
Obtaining tuple embeddings for left table
Traceback (most recent call last):
  File "/srv/splitter.py", line 67, in <module>
    train, test = split_input(str(args.embedding), tableA_df, tableB_df, matches_df, recall=args.recall, top_key=args.top_key, seed=random.randint(0, 4294967295))
  File "/srv/splitter.py", line 39, in split_input
    candidates = generate_candidates(embedding_path, tableA_df, tableB_df, matches_df, top_key=top_key)
  File "/srv/splitter.py", line 22, in generate_candidates
    candidate_set_df = db.block_datasets(tableA_df, tableB_df, cols_to_block)
  File "/srv/DeepBlocker/deep_blocker.py", line 61, in block_datasets
    self.left_tuple_embeddings = self.tuple_embedding_model.get_tuple_embedding(self.left_df["_merged_text"])
  File "/srv/DeepBlocker/tuple_embedding_models.py", line 171, in get_tuple_embedding
    return self.autoencoder_model.get_tuple_embedding(embedding_matrix)
  File "/srv/DeepBlocker/dl_models.py", line 69, in get_tuple_embedding
    return self.encoder(t1).detach().numpy()
  File "/opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/opt/conda/lib/python3.10/site-packages/torch/nn/modules/container.py", line 204, in forward
    input = module(input)
  File "/opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/opt/conda/lib/python3.10/site-packages/torch/nn/modules/linear.py", line 114, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument for argument mat1 in method wrapper_addmm)
```