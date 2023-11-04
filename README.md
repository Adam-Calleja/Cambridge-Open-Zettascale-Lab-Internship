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

To begin my internship I carried out some initial research into HPC and came across the ExaMon M100 dataset which contains a large amount of telemetry data for CINECA’s Marconi100 Tier-0 supercomputer. Since I knew this was similar to the data I would work with during my intership I decided to carry out a short EDA on a small subset of this data, which can be found in the **ExaMon EDA** Jupyter Notebook within the *Introductory-Work* folder.

After completing my inital EDA on the ExaMon M100 dataset, I began working on the internship project. During the majority of the project I used Jupyter Notebooks to organise and annotate my code. I decided to split the work into multiple notebooks to maintain readability and organisation. I will briefly explain each Jupyter Notebook below: 

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

*NOTE: I have removed any code that prepares the Python scripts to query the VictoriaMetrics API as this is a private database. As a reult, the Python scripts that would query VictoriaMetrics will produce an error when they run.*

## Repository Conents
- The **data** folder contains a sample of the data input required by my framework. Due to the sizes of the .csv files I have not included the files created by my Jupyter notebooks. I have also included only the most recent job accounting .csv file. I will describe the data files included below:
    - The **JulyAnonymizedData.csv** file contains anonymised job accounting data for July. This data can be obtained using SLURM's sacct command as decribed above. To keep user data private, this data is anonymized containing hashes.
    - The **partition-info-all** file contains cluster partition data, which can be obtained using SLURM's sinfo command.
- The **notebooks** folder contains all of the Jypiter Notebooks created during this project. 
- The **Introductory-Work** folder contains an EDA project I completed prior to starting the main internship project. 
- The **carbon-framework.py** file contains the Python script which was the main output of this internship project. 
- The **Summer-Internship-Presentation-Slides** pdf conatins the slides I created for my presentation concluding my project. 
- The **Summer Internship Technical Report** pdf contains the technical report I wrote to conclude my prject. 

## Authors and acknowledgment
This project was completed as part of my internship at the Cambridge Open Zettascale Lab, in collaboration with the University of Cambridge.

Many thanks to Projeet Bhaumik, a fellow student Intern who helped with the initial
processing and analysis of the job accounting data and Dominic Friend who mentored me
throughout this summer internship project and was extremely helpful and supportive.
