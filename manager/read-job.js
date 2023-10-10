import { Worker, Job } from 'bullmq';

const connection = {
    host: "localhost",
    port: 6379
};

const worker = new Worker(
    'debugger',
    async (job) => {
        console.log(job);
        // Optionally report some progress
        await job.updateProgress(42);

        // Optionally sending an object as progress
        await job.updateProgress({ foo: 'bar' });

        // Do something with job
        return 'some value';
    },
    { autorun: false, connection },
);

worker.run();