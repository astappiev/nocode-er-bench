import 'process';
import {Queue} from 'bullmq';

const connection = {
    host: "localhost",
    port: 6379
};

const myQueue = new Queue('debugger', {connection});

async function addJobs() {
    await myQueue.add('myJobName', {foo: 'bar'});
    await myQueue.add('myJobName', {qux: 'baz'});
    console.log('Jobs added');
}

await addJobs();
process.exit();
