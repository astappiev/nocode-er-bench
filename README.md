# No-code Benchmarking of Entity Resolution

https://swimlanes.io/u/k3Rmy375P

This is great news! The main requirement is to implement the following process:
    to set several DL-based matching algorithms running on a server as Docker containers
    to feed one of them with three sets of record pairs from pyJedAI' blocking (training, validation and testing), 
    to receive the labels of the pairs in the testing set

We will also need some visualizations, but this should be easy.
The docker images are available here: https://github.com/gpapadis/DLMatchers/tree/main/dockers/mostmatchers
The scenario is described in more detail in Section 3.2 here: https://www.overleaf.com/9862813521fjcqwzqnbmxc
pyJedAI is available here: https://github.com/AI-team-UoA/pyJedAI

I forgot to mention that the dataset is here: https://github.com/gpapadis/DLMatchers/tree/main/EMTransformer/data/abt_buy . The model uses the files train.csv, test.csv and valid.csv.

# To build container, run

docker build -t emtransformer emtransformer
docker run -it --entrypoint=/bin/bash --gpus all  emtransformer

Some more matcher scripts are available here:
https://github.com/nishadi/DLMatchers/tree/main/dockers/mostmatchers/scripts
