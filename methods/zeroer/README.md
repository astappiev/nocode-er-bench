# ZeroER

https://github.com/nishadi/zeroer
entrypoint.py based on zeroer.py

## How to use

You can directly execute the docker image as following:
```bash
docker run --rm -v .:/data zeroer
```
This will assume that you have the input dataset in the current directory,
it will mount it as `/data` and will output the results in the `output` subdirectory.

You can override the input and output directories by providing them as arguments to the docker image:
```bash
docker run -v ../../datasets/d2_abt_buy:/data/input:ro -v ../../test:/data/output zeroer /data/input /data/output
```
