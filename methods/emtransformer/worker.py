import os
import sys
import asyncio
import subprocess
from bullmq import Worker

async def process(job, job_token):
    # job.data will include the data added to the queue
    # TODO: override some params from job.data

    # Job input
    # - epochs
    # -? model_type
    # - data_test
    # - data_train
    # - data_valid

    subprocess.run([
        "python3",
        "/methods/DLMatchers/EMTransformer/src/run_all.py",
        "--model_type='roberta'",
        "--model_name_or_path='roberta-base'",
        "--data_processor=DeepMatcherProcessor",
        "--data_dir=../data/abt_buy",
        "--train_batch_size=16",
        "--eval_batch_size=16",
        "--max_seq_length=180",
        "--num_epochs=15.0",
        "--seed=22",
    ], stderr=sys.stderr, stdout=sys.stdout)

    # Job output
    # - f1
    # -? precision
    # -? recall

async def main():
    queue_server = os.environ['REDIS_SERVER'] # the line should be like "rediss://<user>:<password>@<host>:<port>"
    queue_name = "emtransformer" # we put a model name here, each model will have it's own queue

    worker = Worker(queue_name, process, {"connection": queue_server})

    # This while loop is just for the sake of this example
    # you won't need it in practice.
    while True: # Add some breaking conditions here
        await asyncio.sleep(1)

    # When no need to process more jobs we should close the worker
    await worker.close()

if __name__ == "__main__":
    asyncio.run(main())
