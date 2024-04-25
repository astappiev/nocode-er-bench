import React, {useState, useEffect, useRef} from 'react';
import {DataTable} from 'primereact/datatable';
import {Column} from 'primereact/column';
import {Button} from 'primereact/button';
import {RadioButton} from 'primereact/radiobutton';
import {InputNumber} from 'primereact/inputnumber';
import {FileUpload} from 'primereact/fileupload';
import {Dropdown} from 'primereact/dropdown';
import {InputText} from "primereact/inputtext";
import {ProgressBar} from "primereact/progressbar";
import {ProgressSpinner} from "primereact/progressspinner";

export default function Home() {
  const scenarios = [
    {code: 'filter', name: 'Filtering'},
    {code: 'verify', name: 'Verification'},
    {code: 'progress', name: 'Progressive'},
  ];

  // data
  const [algorithms, setAlgorithms] = useState(null);
  const [datasets, setDatasets] = useState(null);
  const [results, setResults] = useState({});

  // fields
  const [selectedScenario, setSelectedScenario] = useState(scenarios[0]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState(null);
  const [recall, setRecall] = useState(0.85);
  const [epochs, setEpochs] = useState(10);
  const [email, setEmail] = useState('');

  // state
  const [isLoading, setLoading] = useState(true);
  const [formDisabled, setDisabled] = useState(false);
  const [progress, setProgress] = useState(0);
  const [showResults, setShowResults] = useState(false);

  useEffect(() => {
    fetch('/api/algorithms')
      .then((res) => res.json())
      .then((data) => {
        setAlgorithms(data);
        setSelectedAlgorithm(data.find(algo => algo.scenarios.includes(selectedScenario.code)));
        setLoading(!algorithms || !datasets);
      });

    fetch('/api/datasets')
      .then((res) => res.json())
      .then((data) => {
        setDatasets(data);
        setLoading(!algorithms || !datasets);
      });
  }, [])

  const onSubmit = (e) => {
    setDisabled(true);
    setProgress(2);

    const fakeWaitFor = 57_000; // ms
    setTimeout(() => {
      setProgress(5);
    }, fakeWaitFor * 0.1);

    setTimeout(() => {
      setProgress(10);
    }, fakeWaitFor * 0.15);

    setTimeout(() => {
      setProgress(20);
    }, fakeWaitFor * 0.3);

    setTimeout(() => {
      setProgress(80);
    }, fakeWaitFor * 0.85);

    setTimeout(() => {
      setProgress(95);

      fetch('/api/submit', {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          scenario: selectedScenario.code,
          dataset: selectedDataset.code,
          algorithm: selectedAlgorithm.code,
          recall: recall,
          epochs: epochs,
          email: email,
        })
      })
        .then((res) => res.json())
        .then((data) => {
          setResults(data);
          setProgress(100);
          setShowResults(true);
        });
    }, fakeWaitFor);
  }

  if (isLoading) return <p>Loading...</p>
  if (!algorithms || !datasets) return <p>Something went wrong</p>

  return <div>
    <h1 className="text-4xl font-bold">No-code Benchmarking of Entity Resolution</h1>

    <h3 className="mb-2">Select a scenario</h3>
    <div className="flex flex-wrap gap-3">
      {scenarios.map((scenario) => {
        return (
          <div key={scenario.code} className="flex align-items-center">
            <RadioButton inputId={scenario.code} name="scenario" value={scenario}
                         onChange={(e) => setSelectedScenario(e.value)}
                         checked={selectedScenario.code === scenario.code} disabled={formDisabled}/>
            <label htmlFor={scenario.code} className="ml-2">{scenario.name}</label>
          </div>
        );
      })}
    </div>

    <div className="grid mt-4">
      <div className="col">
        <h2>Dataset</h2>

        <div className="flex flex-column gap-2 mb-3">
          <label htmlFor="dataset">Predefined dataset</label>
          <Dropdown id="dataset" aria-describedby="dataset-help"
                    value={selectedDataset} onChange={(e) => setSelectedDataset(e.value)}
                    disabled={formDisabled}
                    options={datasets} optionLabel="name"
                    placeholder="Select a dataset" className="w-full"/>
          <small id="dataset-help">You can select one of the common dataset, or upload your own
            below</small>
        </div>

        <div className="flex flex-column gap-2 mb-3">
          <label htmlFor="dataset_file">Own dataset</label>
          <FileUpload id="dataset_file" aria-describedby="dataset_first-help" mode="basic"
                      name="dataset_file[]" accept="text/*" maxFileSize={1000000}
                      disabled={formDisabled}/>
          <small id="dataset_first-help">Please select the first part</small>

          <FileUpload id="dataset_second" aria-describedby="dataset_second-help" mode="basic"
                      name="dataset_file[]" accept="text/*" maxFileSize={1000000}
                      disabled={formDisabled}/>
          <small id="dataset_second-help">Please select the second part</small>

          <FileUpload id="dataset_ground" aria-describedby="dataset_ground-help" mode="basic"
                      name="dataset_file[]" accept="text/*" maxFileSize={1000000}
                      disabled={formDisabled}/>
          <small id="dataset_ground-help">Ground truth</small>
        </div>
      </div>

      <div className="col">
        <h2>Model parameters</h2>

        <div className="flex flex-column gap-2 mb-3">
          <label htmlFor="algorithm">Algorithm</label>
          <Dropdown id="algorithm" aria-describedby="algorithm-help"
                    value={selectedAlgorithm} onChange={(e) => setSelectedAlgorithm(e.value)}
                    disabled={formDisabled}
                    options={algorithms.filter(algo => algo.scenarios.includes(selectedScenario.code))}
                    optionLabel="name"
                    placeholder="Select a model" className="w-full"/>
          <small id="algorithm-help">Which algorithm / model to use</small>
        </div>

        <div className={(selectedAlgorithm.params.includes('recall') ? null : "hidden")}>
          <div className="flex flex-column gap-2 mb-3">
            <label htmlFor="recall">Recall</label>
            <InputNumber id="recall" aria-describedby="recall-help" className="w-full"
                         value={recall} onValueChange={(e) => setRecall(e.value)}
                         disabled={formDisabled}
                         minFractionDigits={2} min={0} max={1} step={0.05} mode="decimal"
                         showButtons/>
            <small id="recall-help">A recall value between 0 and 1</small>
          </div>
        </div>

        <div className={(selectedAlgorithm.params.includes('epochs') ? null : "hidden")}>
          <div className="flex flex-column gap-2 mb-3">
            <label htmlFor="epochs">Epochs</label>
            <InputNumber id="epochs" aria-describedby="epochs-help" className="w-full"
                         value={epochs} onValueChange={(e) => setEpochs(e.value)}
                         disabled={formDisabled}
                         minFractionDigits={0} min={5} max={50} step={5} showButtons/>
            <small id="epochs-help">A recall value between 0 and 1</small>
          </div>
        </div>
      </div>
    </div>

    <div className="grid mt-4 align-items-center">
      <div className="col">
        <div className="flex flex-column gap-2 mb-3">
          <label htmlFor="email">Email</label>
          <InputText id="email" aria-describedby="email-help" className="w-full"
                     value={email} onChange={(e) => setEmail(e.target.value)}
                     disabled={formDisabled}/>
          <small id="email-help">We will notify when training is complete, it may take time</small>
        </div>
      </div>

      <div className="col">
        <Button label="Submit" onClick={onSubmit}/>
      </div>
    </div>

    <div className={"grid mt-4 " + (progress > 0 && progress < 100 ? null : "hidden")}>
      <ProgressBar value={progress} className="w-full"></ProgressBar>
      <ProgressSpinner className="mt-6 mx-auto"/>
    </div>

    <div className={"mt-4 " + (showResults ? null : "hidden")}>
      <div className="col">
        <h2>Results</h2>
        <DataTable value={results} tableStyle={{minWidth: '50rem'}}>
          <Column field="name" header="Score"/>
          <Column field="value" header="Value"/>
        </DataTable>
      </div>
    </div>
  </div>
}

