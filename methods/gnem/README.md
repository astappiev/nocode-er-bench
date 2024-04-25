
To test the **GNEM** docker, use the following commands:

* docker run -it --gpus all --entrypoint=/bin/bash gnem
* cd /workspace/GNEM
* python train.py --seed 28 --log_freq 5 --lr 0.0001 --embed_lr 0.00002 --epochs 10 --batch_size 2 --tableA_path data/abt_buy/tableA.csv --tableB_path data/abt_buy/tableB.csv --train_path data/abt_buy/train.csv --test_path data/abt_buy/test.csv --val_path data/abt_buy/val.csv --gpu 0 --gcn_layer 1 --test_score_type mean min max
