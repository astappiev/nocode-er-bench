# HierMatcher

https://github.com/nishadi/EntityMatcher

## How to use

You can directly execute the docker image as following:
```bash
docker run --rm -v .:/data hiermatcher
```
This will assume that you have the input dataset in the current directory,
it will mount it as `/data` and will output the results in the `output` subdirectory.

You can override the input and output directories by providing them as arguments to the docker image:
```bash
docker run -v ../../datasets/d2_abt_buy:/data/input:ro -v ../../test:/data/output hiermatcher /data/input /data/output
```
