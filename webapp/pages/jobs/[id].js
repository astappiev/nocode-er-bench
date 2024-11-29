import React, {useState} from 'react';
import {DataTable} from 'primereact/datatable';
import {Column} from 'primereact/column';
import {Button} from 'primereact/button';
import {RadioButton} from 'primereact/radiobutton';
import {InputNumber} from 'primereact/inputnumber';
import {FileUpload} from 'primereact/fileupload';
import {Dropdown} from 'primereact/dropdown';
import {InputText} from "primereact/inputtext";

import prisma from "../../prisma/client";

export const getServerSideProps = async ({query}) => {
  const {id} = query;

  const job = await prisma.job.findFirst({
    where: {id: id},
    include: {
      algorithm: true,
      dataset: true,
      result: true
    }
  });

  return {
    props: {job}
  }
}

export default function Home({job}) {
  if (!job) {
    return <div>
      <h1 className="text-4xl font-bold">Job not found</h1>
    </div>
  }

  return <div>
    <h1 className="text-4xl font-bold">JobId: {job.id}</h1>
    <p className="font-bold">Dataset: {job.dataset.name}</p>
    <p className="font-bold">Algorithm: {job.algorithm.name}</p>
    <p className="font-bold">Recall: {job.recall}</p>
    <p className="font-bold">Epochs: {job.epochs}</p>
    <p className="font-bold text-lg text-primary">Status: {job.status}</p>
  </div>
}

