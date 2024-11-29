// Next.js API route support: https://nextjs.org/docs/api-routes/introduction

import prisma from "../../prisma/client";

export default async function handler(req, res) {
  if (req.method === 'POST') {
    if (!req.body.datasetId || !req.body.algorithmId) {
      res.status(400)
    }

    if (!req.body.force) {
      const existingResults = await prisma.result.findMany({
        where: {
          job: {
            status: 'completed',
            scenario: req.body.scenario,
            datasetId: req.body.datasetId,
            algorithmId: req.body.algorithmId,
            recall: req.body.recall,
            epochs: req.body.epochs,
          }
        }
      });

      if (existingResults.length > 0) {
        return res.status(200).json(existingResults);
      }
    }

    await prisma.job.create({
      data: {
        status: 'pending',
        scenario: req.body.scenario,
        datasetId: req.body.datasetId,
        algorithmId: req.body.algorithmId,
        recall: req.body.recall,
        epochs: req.body.epochs,
        notifyEmail: req.body.notifyEmail,
      }
    });

    res.status(201);
  } else {
    res.status(405)
  }
}
