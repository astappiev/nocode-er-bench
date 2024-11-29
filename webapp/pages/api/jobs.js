// Next.js API route support: https://nextjs.org/docs/api-routes/introduction

import prisma from "../../prisma/client";

export default async function handler(req, res) {
  const jobs = await prisma.job.findMany({
    where: {
      status: 'pending'
    },
    include: {
      algorithm: true,
      dataset: true
    }
  });

  res.status(200).json(jobs);
}
