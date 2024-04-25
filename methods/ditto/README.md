To test the **DITTO** docker, use the following commands:

*  cd /workspace/ditto
*  CUDA_VISIBLE_DEVICES=0 python train_ditto.py  --task Structured/Beer --batch_size 64  --max_len 256  --lr 3e-5  --n_epochs ${EPOCHS} --lm roberta  --fp16 --da del --dk product  --summarize
