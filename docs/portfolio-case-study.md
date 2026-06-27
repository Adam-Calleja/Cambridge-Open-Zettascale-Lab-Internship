# Portfolio Case Study: CSD3 HPC Carbon Accounting Framework

## Summary

This project built a Python framework for estimating the energy consumption and carbon footprint of jobs running on the University of Cambridge CSD3 supercomputer. The work connected SLURM job-accounting records, cluster partition metadata, CSD3 node-level power telemetry, and UK grid carbon intensity data to produce per-job carbon estimates.

The public repository is an anonymised portfolio artifact. It preserves the code, methodology, sample schema, report, and presentation, while excluding private job records and private telemetry access.

## Problem

HPC centres can usually observe cluster-level power draw, but that does not directly answer a more useful operational question: which jobs, users, workloads, or partitions are responsible for energy use and emissions?

The project aimed to move from aggregate visibility to per-job attribution by estimating:

- total job energy consumption in Wh,
- job carbon footprint in gCO2,
- an equivalent distance driven by a medium-sized diesel car.

## My Role

I developed the analysis and framework during an internship with the Cambridge Open Zettascale Lab. The work included:

- exploring SLURM accounting records and relevant HPC telemetry concepts,
- cleaning and preparing monthly job-accounting data,
- defining which jobs could be attributed reliably,
- implementing the carbon-estimation workflow in Python,
- producing analysis notebooks, a technical report, and final presentation materials.

## Technical Approach

The framework follows a staged data pipeline:

1. Load SLURM accounting records and partition metadata.
2. Remove jobs that cannot be analysed cleanly, such as jobs with zero CPU time, unknown end times, or excluded partitions.
3. Normalise timestamp and numeric columns.
4. Identify exclusive jobs by comparing allocated CPUs against partition CPU counts.
5. Expand node-list ranges and check temporal overlap to catch jobs that appear non-exclusive by CPU count but do not actually share nodes during runtime.
6. Query CSD3 VictoriaMetrics power telemetry for the nodes used by each job.
7. Integrate power readings over time to estimate energy consumption.
8. Join 30-minute UK Carbon Intensity API data.
9. Write per-job energy, carbon, and equivalent-distance outputs.

## Engineering Decisions

The most important modelling decision was to focus on exclusive jobs: jobs that do not share their nodes with other jobs during runtime. This makes attribution defensible because observed node power can be assigned to a single job without splitting energy between unrelated workloads.

The framework also records jobs that are shared only with the same user, which points to a natural next extension: attributing same-user shared-node workloads as a group.

Private infrastructure access was isolated behind configuration. The public script no longer includes VictoriaMetrics endpoint details and instead expects callers to supply access through environment variables.

## Public Artifact Boundaries

The original analysis used private Cambridge operational data and private CSD3 telemetry access. Those cannot be published. This repository therefore includes:

- the framework code,
- the methodology notebooks,
- anonymised sample job-accounting data,
- public partition metadata,
- the technical report,
- the presentation slides.

It does not include:

- full monthly SLURM job-accounting datasets,
- private user, account, or job-name identifiers,
- VictoriaMetrics endpoint, proxy, or authentication details,
- the ability to reproduce the original end-to-end run outside the Cambridge environment.

## Impact

The project produced a working first framework for attributing CSD3 energy use and carbon emissions to individual jobs and users, rather than only reasoning about whole-cluster power consumption.

Headline findings from the original analysis included:

- exclusive jobs accounted for 68.7% of July cluster usage,
- around 41% of remaining shared usage involved jobs submitted by the same user,
- per-job attribution is increasingly relevant as HPC systems move toward multi-megawatt pre-exascale deployments.

## What to Inspect

- [`carbon-framework.py`](../carbon-framework.py): the consolidated framework script.
- [`notebooks/Slurm-EDA-June-Exclusive.ipynb`](../notebooks/Slurm-EDA-June-Exclusive.ipynb): development of the exclusive-job attribution logic.
- [`notebooks/carbon-footprint-july.ipynb`](../notebooks/carbon-footprint-july.ipynb): notebook implementation of the carbon workflow.
- [`data/JulyAnonymizedData-sample.csv`](../data/JulyAnonymizedData-sample.csv): anonymised sample of the expected SLURM accounting schema.
- [`Summer Internship Technical Report.pdf`](../Summer%20Internship%20Technical%20Report.pdf): full methodology and results.

## Follow-On Work

The clearest next steps would be:

- package the framework as a CLI with explicit input validation,
- add unit tests around node-list expansion and overlap detection,
- support attribution for same-user shared-node jobs,
- cache external carbon-intensity responses more deliberately,
- produce a small synthetic end-to-end fixture that can run without private telemetry.
