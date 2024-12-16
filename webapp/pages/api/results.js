import prisma from "../../prisma/client";

export default async function handler(req, res) {
  if (req.method === 'POST') {

    if (!req.body.jobId || !req.body.status || (req.body.status !== 'completed' || req.body.status !== 'failed') || !req.body.elapsed) {
      res.status(400)
    }

    await prisma.result.create({
      data: {
        jobId: req.body.jobId,

        f1: req.body.f1,
        precision: req.body.precision,
        recall: req.body.recall,
        maxMem: req.body.maxMem,
        elapsed: req.body.elapsed,
      }
    });

    await prisma.job.update({
      where: {
        id: req.body.jobId
      },
      data: {
        status: req.body.status
      }
    });

    res.status(200);
  } else {
    res.status(405)
  }
}
