# CSD3 HPC Carbon Accounting Framework

Python framework for estimating per-job energy use and carbon emissions from SLURM accounting data on the University of Cambridge CSD3 supercomputer.

This repository is the public, anonymised portfolio version of work completed during an internship with the Cambridge Open Zettascale Lab. It is framed as an inspectable engineering artifact: the code, methodology, sample data, report, and presentation are included, while private SLURM records and VictoriaMetrics telemetry access are intentionally excluded.

## At a Glance

| Area | Detail |
| --- | --- |
| Problem | Attribute HPC energy use and carbon emissions back to individual jobs rather than only reporting cluster-level totals. |
| Core artifact | `carbon-framework.py`, a Python pipeline that filters SLURM accounting data, identifies exclusive jobs, queries node power telemetry, joins UK carbon intensity data, and writes per-job carbon outputs. |
| Public data | A 2,000-row anonymised sample of CSD3-style SLURM accounting data plus partition metadata. |
| Original analysis scale | SLURM accounting data for over 2 million jobs. |
| Stack | Python, pandas, NumPy, requests, joblib, matplotlib, Plotly, Jupyter. |
| Constraints | Full monthly job-accounting data and CSD3 VictoriaMetrics access are private, so the public repo documents the complete method but cannot reproduce the original end-to-end run outside the Cambridge environment. |

## What This Demonstrates

- Data engineering over messy operational HPC records: cleaning, type normalisation, filtering invalid jobs, and joining partition metadata.
- Domain modelling: translating SLURM allocations, node lists, overlap windows, power telemetry, and carbon intensity into per-job estimates.
- Practical privacy handling: publishing only anonymised sample records and removing private telemetry endpoint details.
- Scientific communication: the repository includes the technical report and presentation used to communicate methodology, assumptions, and findings.
- Engineering judgement under constraints: the public artifact preserves the inspectable code path while making the private-data boundary explicit.

## Where to Inspect the Code

| Path | What to look for |
| --- | --- |
| [`carbon-framework.py`](carbon-framework.py) | Main framework: SLURM ingestion, data cleaning, exclusive-job detection, VictoriaMetrics query construction, energy integration, Carbon Intensity API join, and final CSV output. |
| [`notebooks/Slurm-EDA-June-Preparation.ipynb`](notebooks/Slurm-EDA-June-Preparation.ipynb) | Data preparation steps for anonymised monthly SLURM accounting data. |
| [`notebooks/Slurm-EDA-June-Exclusive.ipynb`](notebooks/Slurm-EDA-June-Exclusive.ipynb) | Exploratory development of the exclusive-job logic used to decide which jobs can be attributed cleanly. |
| [`notebooks/Slurm-EDA-June-Analysis.ipynb`](notebooks/Slurm-EDA-June-Analysis.ipynb) | Analysis notebooks for cluster usage, job characteristics, and supporting plots. |
| [`notebooks/carbon-footprint-july.ipynb`](notebooks/carbon-footprint-july.ipynb) | Notebook version of the carbon-footprint workflow before it was consolidated into the framework script. |
| [`data/JulyAnonymizedData-sample.csv`](data/JulyAnonymizedData-sample.csv) | Public sample of the job-accounting schema expected by the framework. |
| [`docs/portfolio-case-study.md`](docs/portfolio-case-study.md) | Hiring-engineer case study: problem, approach, tradeoffs, impact, and public artifact boundaries. |
| [`Summer Internship Technical Report.pdf`](Summer%20Internship%20Technical%20Report.pdf) | Full write-up of methodology, assumptions, and results. |

## Suggested Review Path

1. Read this README for the project framing and public-data boundaries.
2. Skim [`docs/portfolio-case-study.md`](docs/portfolio-case-study.md) for the engineering narrative.
3. Inspect [`carbon-framework.py`](carbon-framework.py), especially the data preparation, exclusivity checks, `getJobPower`, `getJobEnergy`, and `getCarbonFootprint`.
4. Open the technical report if you want the full internship methodology and results.

## Public Data and Reproducibility Boundaries

The public repository is designed for code and methodology inspection, not full external reproduction of the original Cambridge run.

- The included CSV is an anonymised sample, not the full monthly dataset.
- User, account, and job-name identifiers are replaced with synthetic IDs.
- CSD3 VictoriaMetrics endpoint details, proxy configuration, and authentication are not included.
- `carbon-framework.py` now expects VictoriaMetrics access to be supplied via environment variables and raises a clear error if that private dependency is not configured.

The framework expects:

- a SLURM `sacct` CSV using:

```text
-XP --format Account,AllocCPUs,AllocNodes,ElapsedRaw,GID,JobIDRaw,JobName,NodeList,UID,Start,End,Partition
```

- a SLURM `sinfo` partition file using:

```text
--format '%R|%D|%c|'
```

The output CSV contains the estimated energy consumption in Wh, carbon footprint in gCO2, and equivalent distance driven by a medium-sized car in km for each qualifying exclusive job.

## Results & key findings
The project delivered a working Python framework (`carbon-framework.py`) that estimates, for each qualifying job on CSD3, its **energy consumption (Wh)**, **carbon footprint (gCO₂)**, and the **equivalent distance driven by a medium-sized diesel car (km)** — combining per-node power telemetry (VictoriaMetrics) with the public [Carbon Intensity API](https://carbonintensity.org.uk). It was designed to make energy use and carbon emissions attributable to individual jobs and users, rather than only visible as aggregate cluster-level consumption.

Headline findings from the analysis of SLURM job-accounting data for **over 2 million jobs**:

- **68.7%** of cluster usage in July was made up of *exclusive* jobs (jobs that do not share a node with any other job) — all exclusive by CPU count.
- Of the remaining *shared* usage, around **41%** consisted of jobs submitted by the same user — the natural next group to fold into the model.
- For context, the Cambridge HPC system draws **over 1 MW continuously**, with forthcoming pre-exascale systems projected at roughly **7 MW** — underlining why per-job energy visibility matters.

Full methodology, assumptions, and results are in the **[Technical Report](Summer%20Internship%20Technical%20Report.pdf)** and the **[presentation slides](Summer-Internship-Presentation-Slides.pdf)**.

## Repository Contents
- The **data** folder contains a small, anonymised sample of the job-accounting data used by the framework. The full dataset is not published.
    - **JulyAnonymizedData-sample.csv** contains a 2,000-row sample of the July job-accounting data, of the kind obtained using SLURM's `sacct` command as described above. **All personal identifiers have been removed:** the user, account, and job-name fields are replaced with consistent synthetic IDs (e.g. `user_0001`, `acct_0001`, `job_0001`), and cancelling-user UIDs recorded in the job state are redacted. Node names and partition data are retained because they describe cluster infrastructure rather than individuals.
    - **partition-info-all** contains cluster partition data, which can be obtained using SLURM's `sinfo` command.
- The **docs** folder contains the portfolio case study.
- The **notebooks** folder contains the exploratory and methodology notebooks behind the final framework.
- The **Introductory-Work** folder contains the preliminary ExaMon EDA completed before the main CSD3 analysis.
- **carbon-framework.py** contains the consolidated Python framework.
- **Summer-Internship-Presentation-Slides.pdf** contains the final project presentation.
- **Summer Internship Technical Report.pdf** contains the full technical report.

## Running the code
**The notebooks are included as annotated methodology, not as a runnable pipeline.** They were developed inside the Cambridge Research Computing network and read a chain of intermediate data files (raw monthly SLURM accounting data and several processed CSVs) that are **not included in this repository**, for size and privacy reasons. `carbon-footprint-july.ipynb` additionally requires access to the CSD3 VictoriaMetrics power telemetry, which is only reachable from within the Cambridge network. They therefore will not execute end-to-end outside that environment, and are provided to document the approach and the code.

`carbon-framework.py` runs on a job-accounting CSV of the form in `data/JulyAnonymizedData-sample.csv`. The exclusive-job processing runs locally, but the energy/carbon calculation queries VictoriaMetrics for per-node power; without access it raises a clear, descriptive error. See the configuration block at the top of the script (`VICTORIA_METRICS_URL` and related environment variables).

## Authors and acknowledgment
This project was completed as part of my internship at the Cambridge Open Zettascale Lab, in collaboration with the University of Cambridge.

Many thanks to Projeet Bhaumik, a fellow student Intern who helped with the initial
processing and analysis of the job accounting data and Dominic Friend who mentored me
throughout this summer internship project and was extremely helpful and supportive.
