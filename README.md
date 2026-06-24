# Cambridge-Open-Zettascale-Lab-Internship

## Description
This repository contains all of the work I completed during my internship at the University of Cambridge, in partnership with the Cambridge Open Zettascale Lab.

The objective of this summer internship project was to develop a framework to calculate and report the energy consumption and carbon footprint of jobs running on the CSD3 supercomputer. I decided to implement this framework in Python, utilising the following Python libraries: 

- pandas
- numpy
- sys
- datetime
- re
- requests
- pytz
- joblib
- matplotlib
- plotly

To begin my internship I carried out some initial research into HPC and came across the ExaMon M100 dataset which contains a large amount of telemetry data for CINECA’s Marconi100 Tier-0 supercomputer. Since I knew this was similar to the data I would work with during my internship I decided to carry out a short EDA on a small subset of this data, which can be found in the **ExaMon EDA** Jupyter Notebook within the *Introductory-Work* folder.

After completing my initial EDA on the ExaMon M100 dataset, I began working on the internship project. During the majority of the project I used Jupyter Notebooks to organise and annotate my code. I decided to split the work into multiple notebooks to maintain readability and organisation. I will briefly explain each Jupyter Notebook below: 

- **Slurm-EDA-Sample-Data**

    In this Jupyter Notebook I start an EDA on SLURM job accounting data for the CSD3 supercomputer. While starting this EDA I did not have access to users' job accounting data so I created some sample data by submitting my own jobs on the CSD3 supercomputer using the SLURM job scheduler. The purpose of this Jupyter Notebook was to familiarise myself with SLURM and using the cluster. I also created a DataFrame containing information about each partition of the cluster which I access in later notebooks. 

- **Slurm-EDA-June-Preparation**

    In this Jupyter Notebook I had access to anonymised SLURM job accounting data for all jobs run on CSD3 during June. I began to process and clean the data by removing unnecessary rows. I also ensured all columns had the correct data type.

- **Slurm-EDA-June-Exclusive**

    In this Jupyter Notebook I continued the data processing started in the *Slurm-EDA-June-Preparation* notebook. In particular I identified which jobs are exclusive (which jobs do not share the node they run on with any other jobs during their runtime). This is important since, for the sake of this project, we assume that all jobs are exclusive and remove any jobs that are not.

- **Slurm-EDA-June-Analysis**

    In this Jupyter Notebook, I continued the EDA started in the *Slurm-EDA-Sample-Data* notebook, analysing the data by calculating statistics and creating plots using the *matplotlib* and *plotly* libraries.

- **carbon-footprint-july**

    In this final Jupyter Notebook I expanded on the previous notebooks to calculate the energy consumption and carbon footprints of exclusive jobs running on the CSD3 supercomputer. In this notebook I used a different dataset containing all jobs running on CSD3 during July. I first used the previous notebooks to preprocess this new dataset. 

After writing the code in the *carbon-footprint-july* notebook, I had code across all of my notebooks that could calculate the energy consumption and carbon footprint of jobs given a dataset of job accounting data. I then wrote the **carbon-framework.py** Python script which takes in the paths to the necessary data files as an argument and creates a new CSV file containing the energy consumption and carbon footprint of exclusive jobs in the input data.

This script takes in three file paths as parameters: 

- the path to the CSV file containing job accounting data,
- the path to the CSV file containing cluster partition data, 
- the path of the output CSV file which is created by the script.

The first CSV file can be obtained by using SLURM's sacct command with the following settings: 

    -XP --format Account,AllocCPUs,
    AllocNodes,ElapsedRaw,GID,JobIDRaw,JobName,NodeList,UID,Start,End,Partition

The second CSV file can obtained by using SLURM's sinfo command with the following settings: 

    --format '%R|%D|%c|'

The output CSV file contains the energy consumption, in Wh; the carbon footprint, in gCO2; and the distance driven by a medium sized car to produce the same carbon footprint, in km, for each exclusive job in the input job accounting CSV file.

*NOTE: During this project, we only calculated the energy consumption and carbon footprints of exclusive jobs. We define an exclusive job to be **any job which does not share the node it runs on with any other jobs during its runtime**.*

*NOTE: I have removed any code that prepares the Python scripts to query the VictoriaMetrics API as this is a private database. As a result, the Python scripts that would query VictoriaMetrics will produce an error when they run.*

## Results & key findings
The project delivered a working Python framework (`carbon-framework.py`) that estimates, for each qualifying job on CSD3, its **energy consumption (Wh)**, **carbon footprint (gCO₂)**, and the **equivalent distance driven by a medium-sized diesel car (km)** — combining per-node power telemetry (VictoriaMetrics) with the public [Carbon Intensity API](https://carbonintensity.org.uk). It was the Research Computing Service's first means of attributing energy use and carbon emissions back to individual jobs and users.

Headline findings from the analysis of SLURM job-accounting data for **over 2 million jobs**:

- **68.7%** of cluster usage in July was made up of *exclusive* jobs (jobs that do not share a node with any other job) — all exclusive by CPU count.
- Of the remaining *shared* usage, around **41%** consisted of jobs submitted by the same user — the natural next group to fold into the model.
- For context, the Cambridge HPC system draws **over 1 MW continuously**, with forthcoming pre-exascale systems projected at roughly **7 MW** — underlining why per-job energy visibility matters.

Full methodology, assumptions, and results are in the **[Technical Report](Summer%20Internship%20Technical%20Report.pdf)** and the **[presentation slides](Summer-Internship-Presentation-Slides.pdf)**.

## Repository Contents
- The **data** folder contains a small, anonymised sample of the job accounting data used by my framework (the full dataset is not published). I will describe the data files included below:
    - The **JulyAnonymizedData-sample.csv** file contains a 2,000-row sample of the July job accounting data, of the kind obtained using SLURM's `sacct` command as described above. **All personal identifiers have been removed:** the user, account, and job-name fields are replaced with consistent synthetic IDs (e.g. `user_0001`, `acct_0001`, `job_0001`), and cancelling-user UIDs recorded in the job state are redacted. Node names and partition data are retained, as they describe public cluster infrastructure rather than individuals.
    - The **partition-info-all** file contains cluster partition data, which can be obtained using SLURM's sinfo command.
- The **notebooks** folder contains all of the Jupyter Notebooks created during this project. 
- The **Introductory-Work** folder contains an EDA project I completed prior to starting the main internship project. 
- The **carbon-framework.py** file contains the Python script which was the main output of this internship project. 
- The **Summer-Internship-Presentation-Slides** pdf contains the slides I created for my presentation concluding my project. 
- The **Summer Internship Technical Report** pdf contains the technical report I wrote to conclude my project. 

## Running the code
**The notebooks are included as annotated methodology, not as a runnable pipeline.** They were developed inside the Cambridge Research Computing network and read a chain of intermediate data files (raw monthly SLURM accounting data and several processed CSVs) that are **not included in this repository**, for size and privacy reasons. `carbon-footprint-july.ipynb` additionally requires access to the CSD3 VictoriaMetrics power telemetry, which is only reachable from within the Cambridge network. They therefore will not execute end-to-end outside that environment, and are provided to document the approach and the code.

`carbon-framework.py` runs on a job-accounting CSV of the form in `data/JulyAnonymizedData-sample.csv`. The exclusive-job processing runs locally, but the energy/carbon calculation queries VictoriaMetrics for per-node power; without access it raises a clear, descriptive error. See the configuration block at the top of the script (`VICTORIA_METRICS_URL` and related environment variables).

## Authors and acknowledgment
This project was completed as part of my internship at the Cambridge Open Zettascale Lab, in collaboration with the University of Cambridge.

Many thanks to Projeet Bhaumik, a fellow student Intern who helped with the initial
processing and analysis of the job accounting data and Dominic Friend who mentored me
throughout this summer internship project and was extremely helpful and supportive.
