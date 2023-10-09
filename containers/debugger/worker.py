import os
import asyncio
import subprocess
from bullmq import Worker, Job


async def process(job: Job, job_token):
    # job.data will include the data added to the queue
    # TODO: override some params from job.data

    print(job)
    print(job.data)

    subprocess.run(['/usr/bin/python3', '-c', 'from datetime import date; print(date.today())'])


async def main():
    queue_server = os.environ['REDIS_SERVER']  # the line should be like "rediss://<user>:<password>@<host>:<port>"
    queue_name = "debugger"                    # we put a model name here, each model will have it's own queue

    print("Starting worker...")
    print(f"Queue server: {queue_server}")
    worker = Worker(queue_name, process, {"connection": queue_server})
    print("Worker initialized.")

    # This while loop is just for the sake of this example
    # you won't need it in practice.
    while True:  # Add some breaking conditions here
        await asyncio.sleep(1)

    # When no need to process more jobs we should close the worker
    await worker.close()


if __name__ == "__main__":
    print("Hi, I'm the debugger worker.")
    asyncio.run(main())
