# Magellan

https://github.com/anhaidgroup/py_entitymatching

The docker image should be self-contained and should be able to run the method without any additional user input.
The image should contain the source code of the method and input + output transformers.

## Structure
- [Dockerfile](Dockerfile) should contain the instructions to build the docker image.
- [transform.py](transform.py) contains pre-processing and post-processing functions that will be applied to the input and output data.
- [entrypoint.py](entrypoint.py) should contain the main method that will be executed inside the docker container.

## How to use

You can directly execute the docker image as following:
```bash
docker run --rm --mount type=tmpfs,destination=/tmpdir -v .:/data magellan
```
This will assume that you have the input dataset in the current directory,
it will mount it as `/data` and will output the results in the `output` subdirectory.

You can override the input and output directories by providing them as arguments to the docker image:
```bash
docker run --mount type=tmpfs,destination=/tmpdir -v ../../datasets/d2_abt_buy:/data/input:ro -v ../../test:/data/output magellan /data/input /data/output
```
