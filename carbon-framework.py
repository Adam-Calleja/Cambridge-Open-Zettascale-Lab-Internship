"""
carbon-framework.py

Calculates the energy consumption and carbon footprint of jobs running on csd3.

Takes in two .csv file paths as parameters, the first containing job accounting data from
SLURM and the second containing cluster partition data from SLURM. Calculates the 
energy consumption, in Wh; the carbon footprint, in gCO2; and the distance driven by 
a car to produce the same carbon footprint, in km; and stores these values in a .csv f-ile 
named after the month most of the Jobs started running.

Usage:
----------
python carbon-framework.py arg1 arg2 arg3

Parameters:
----------
arg1: string 
    Path to the job accounting data CSV file. This CSV file should be obtained using 
    SLURM's sacct command with the following settings: -XP --format Account,AllocCPUs,
    AllocNodes,ElapsedRaw,GID,JobIDRaw,JobName,NodeList,UID,Start,End,Partition
arg2: string
    Path to the partition data CSV file. This CSV file should be obtained by using
    SLURM's sinfo command with the following settings: --format '%R|%D|%c|'
arg3: string
    The file path of the final CSV file. This CSV file is created by the script and 
    contains all of the job's carbon footprint data as well as the original job accounting 
    data.

Returns:
----------
None 

Example:
----------
python carbon-framework.py sacct-june.csv partition-june.csv

Notes:
----------
For code containing more detailed comments please see the Jupyter Notebooks in 
the following GitLab repository: 
https://gitlab.developers.cam.ac.uk/ac2650/csd3-summer-internship/-/tree/Carbon-Footprint-Adam

"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import requests
import pytz
from joblib import Parallel, delayed

print('start')

# Below we disable the setting with copy warning from pandas. 
pd.options.mode.chained_assignment = None

# We will first read in the job accounting data and the partition data and store 
# these datasets as DataFrames. 
sSlurmDataPath = sys.argv[1]
dfSacct = pd.read_csv(sSlurmDataPath, index_col=0)

sPartitionDataPath = sys.argv[2]
dfPartition = pd.read_csv(sPartitionDataPath, sep='|')

# We will now prepare the two DataFrames above to be used later. 
dfSacct.set_index(['JobIDRaw'], drop=True, inplace=True)

dfPartition = dfPartition.iloc[:, :-1]
dfPartition.index = dfPartition['PARTITION']
dfPartition = dfPartition.drop(['NODES', 'PARTITION'], axis=1)
bNoPlusMask = ~dfPartition['CPUS'].str.endswith('+')
dfPartition = dfPartition[bNoPlusMask]
dfPartition['CPUS'] = pd.to_numeric(dfPartition['CPUS'], errors='coerce')

# We will now process the job accounting data to remove any jobs that do not 
# align with our assumptions.
bNoCPUTimeMask = dfSacct['CPUTimeRAW'] != 0
dfSacct = dfSacct[bNoCPUTimeMask]
bKnownEndTime = dfSacct['End'] != 'Unknown'
dfSacct = dfSacct[bKnownEndTime]
bExcludePartitionMask = dfSacct['Partition'].isin(dfPartition.index)
dfSacct = dfSacct[bExcludePartitionMask]

# We will now ensure that all columns are of the correct type and we will sort the 
# job accoutning data by start time. 
dfSacct['Start'] = pd.to_datetime(dfSacct['Start'], format='%Y-%m-%dT%H:%M:%S')
dfSacct['End'] = pd.to_datetime(dfSacct['End'], format='%Y-%m-%dT%H:%M:%S')

dfSacct.sort_values('Start', axis=0, inplace=True)

# We are now going to check whether jobs are exclusive by our first definition
# of exclusiveness. 
SPartitionNames = dfSacct['Partition']
ICPUsPerNode = dfPartition.loc[SPartitionNames]['CPUS'] 
IAllocatedCPUs = dfSacct['AllocCPUS']
IAllocatedNodes = dfSacct['AllocNodes']

IExclusiveCPUCount = ICPUsPerNode.values * IAllocatedNodes

dfSacct['ExclusiveCPU'] = (IAllocatedCPUs == IExclusiveCPUCount)

# We will now check whether jobs are exclusive by our second definition 
# of exclusiveness. 
bExlusiveMask = dfSacct['ExclusiveCPU'] == False
dfNonExclusive = dfSacct[bExlusiveMask]

dfNonExclusive['NodePrefix'] = dfNonExclusive['NodeList'].str.extract(r'(.+?)(?:-\[|\-)(\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*)(?:\]|$)')[0]
dfNonExclusive['NodeNumbers'] = dfNonExclusive['NodeList'].str.extract(r'(.+?)(?:-\[|\-)(\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*)(?:\]|$)')[1]

df_duplicate = dfNonExclusive.assign(list=dfNonExclusive['NodeNumbers'].str.split(',')).explode('list')
df_duplicate = df_duplicate.assign(consecutive=df_duplicate['list'].str.split('-'))
df_duplicate['consecutive'] = df_duplicate['consecutive'].apply(lambda lsRange: [lsRange] if type(lsRange) == float else list(range(int(lsRange[0]), int(lsRange[1]) + 1)) if len(lsRange) > 1 else [int(lsRange[0])])
df_duplicate = df_duplicate.explode('consecutive')
df_duplicate['NodeList'] = df_duplicate.apply(lambda row: str(row['NodePrefix']) + '-' + str(row['consecutive']), axis=1)
df_duplicate = df_duplicate.drop(['NodePrefix', 'list', 'NodeNumbers', 'consecutive'], axis=1)

df_duplicate.sort_values(['NodeList', 'Start'], axis=0, inplace=True)
df_duplicate_shift = df_duplicate.shift(periods=1)
lPossibleOverlapNodes = df_duplicate[(df_duplicate_shift['NodeList'] == df_duplicate['NodeList']) & (df_duplicate_shift['End'] > df_duplicate['Start'])]['NodeList'].unique()

bPossibleSharedMask = df_duplicate['NodeList'].isin(lPossibleOverlapNodes)
dfPossibleSharedExpanded = df_duplicate[bPossibleSharedMask]

def fFindSharedJobs(job, df):
    """
    Finds all jobs that overlap with the given job on the same node. 

    Compares a given job to all other jobs that start at most 36 hours before the job 
    starts or that start during the job's runtime and returns a list of the job IDs of 
    all the jobs that it overlaps with. 

    Parameters
    ----------
    job: pd.Series
        The pd Series containing all of a job's accounting data from the job accounting
        DataFrame. 
    df: pd.DataFrame
        The pd DataFrame containing all of the job accounting data from SLURM.
    
    Returns:
    ----------
    lOverlapIDs: list
        A list containing the job IDs of all the jobs that overlap with the given job. 

    """
    
    sNode = job.NodeList
    tJobStart = job.Start
    tJobEnd = job.End
    tJobTime = job.ElapsedRaw
    

    bTimeMask = (df.Start <= job.Start + timedelta(seconds = tJobTime)) & (df.Start >= job.Start - timedelta(hours = 36))
    
    StOtherStart = df.Start[bTimeMask]
    StOtherEnd = df.End[bTimeMask]
    
    bTest1 = (df[bTimeMask].index != job.name) 
    bTest2 = (df[bTimeMask]['NodeList'] == sNode)
    bTest3 = ((StOtherStart >= tJobStart) & (StOtherStart <= tJobEnd)) 
    bTest4 = ((StOtherEnd >= tJobStart) & (StOtherEnd <= tJobEnd)) 
    bTest5 = ((StOtherStart <= tJobStart) & (StOtherEnd >= tJobEnd))
    
    bMask = bTest1 & bTest2 & (bTest3 | bTest4 | bTest5)

    lOverlapIDs = list(df[bTimeMask][bMask].index)

    return lOverlapIDs

def sharedSameUser(job, df):
    """
    Checks if a shared job only overlaps with jobs ran by the same user. 

    Parameters
    ----------
    job: pd.Series
        The pd Series containing all of a job's accounting data from the job accounting
        DataFrame. 
    df: pd.DataFrame
        The pd DataFrame containing all of the job accounting data from SLURM.

    Returns
    ----------
    booleana: True if the job only overlaps with jobs ran by the same user, 
              false otherwise. 
    
    """
    
    sUser = job.User
    lOverlapping = job.Overlapping

    lUsers = []

    for sJob in lOverlapping:
        lUsers.append(df.loc[sJob, 'User'])
    
    if len(set(lUsers)) == 1:
        return True
    else: 
        return False
    

lNodes = dfPossibleSharedExpanded['NodeList'].unique()

lNodes = np.delete(lNodes, -1)

lNodeDataFrames =  []

for sNode in lNodes:
    lNodeDataFrames.append(dfPossibleSharedExpanded[dfPossibleSharedExpanded['NodeList'] == sNode].copy())


def fFindOnNode(df):
    df['Overlapping'] = df.apply(lambda row : fFindSharedJobs(row, df), axis=1)
    df['SharedSameUser'] = df.apply(lambda row : sharedSameUser(row, df), axis=1)
    return df

lOverlapDataFrames = Parallel(n_jobs=8)(delayed(fFindOnNode)(df) for df in lNodeDataFrames)

dfPossibleSharedOverlapping = pd.concat(lOverlapDataFrames)

# We will now create two new columns in the job accounting DataFrame, one containing 
# a boolean value representing whether or not the job is exclusive by the second definition
# of exclusiveness, and one containing a boolean value representing whether the job is 
# exclusive by either one of the two definitions.  
bNoOverlap = dfPossibleSharedOverlapping['Overlapping'] == False
dfNoOverlap = dfPossibleSharedOverlapping[bNoOverlap]

lNotOverlapping = list(dfNoOverlap.index.unique())

dfSacct['ExclusiveOverlapping'] = False
dfSacct.loc[lNotOverlapping, 'ExclusiveOverlapping'] = True

def isExclusive(job):
    """
    Checks whether a job is exclusive by either one of our definitions.

    Parameters:
    ----------
    job: pd.Series
        The pd Series containing all of a job's accounting data from the job accounting
        DataFrame.

    Returns:
    ----------
    boolean: True if the job is exclusive by either one of our definitions, 
             False otherwise. 

    """

    bCPU = job.ExclusiveCPU
    bOverlapping = job.ExclusiveOverlapping

    return bCPU or bOverlapping

dfSacct['Exclusive'] = dfSacct.apply(lambda row : isExclusive(row), axis=1)

# We will now add the 'SharedSameUser' column to the job accounting DataFrame.
bSameUser = dfPossibleSharedOverlapping['SharedSameUser'] == True
dfSameUser = dfPossibleSharedOverlapping[bSameUser]

lSameUserIndex = list(dfSameUser.index.unique())

dfSacct['SharedSameUser'] = False
dfSacct.loc[lSameUserIndex, 'SharedSameUser'] = True

# We are now going to define the functions necessary to calculate the carbon 
# footprint of the jobs. 

def dfToUTC(df):
    """
    Returns a pd DataFrame containing columns for the start and end times in UTC.

    Parameters
    ----------
    df: pdDataFrame 
        The pd DataFrame containing all of the job data. This DataFrame must contain 
        a 'Start' and 'End' column of pd DateTime64 objects. 
    
    Returns
    ----------
    df: pdDataFrame
        The pd DataFrame that was passed in as a parameter with two new columns:
        'StartUTC' and 'EndUTC' of pd DateTime64 objects, which contain the original 
        start and end times in UTC rather than local time. 
    """

    df['UTCStart'] = df['Start'].dt.tz_localize('Europe/London')
    df['UTCEnd'] = df['End'].dt.tz_localize('Europe/London')

    df['UTCStart'] = df['UTCStart'].dt.tz_convert(pytz.utc)
    df['UTCEnd'] = df['UTCEnd'].dt.tz_convert(pytz.utc)

    return df 

def getJobPower(jobID, dfJobs):
    """
    Returns a DataFrame containing the power readings, in W, on each node that the job runs on for the duration of the job. 

    Returns a DataFrame whose index is the timestamp and whose columns are the power readings, in W, for each node the job 
    runs on. If there is a problem while querying victoria metrics, an exception is thrown. 

    Parameters
    ----------
    jobID: integer
        The integer job ID for the job in question. 
    dfJobs: pdDataFrame
        The DataFrame containing all of the job data for the time period in question. 
    
    Returns
    ----------
    dfJobPower: pdDataFrame
        The DataFrame containing the power readings, in W, for each node the job runs on. 
    None: NoneType
        Returns None if there is a problem while querying Victoria Metrics
    """

    if dfJobs.index.value_counts().loc[jobID] > 1:
        sStart = dfJobs.loc[jobID, 'UTCStart'].iloc[0].strftime("%Y-%m-%dT%H:%M:%SZ")
        sEnd = dfJobs.loc[jobID, 'UTCEnd'].iloc[0].strftime("%Y-%m-%dT%H:%M:%SZ")
    else: 
        sStart = dfJobs.loc[jobID, 'UTCStart'].strftime("%Y-%m-%dT%H:%M:%SZ")
        sEnd = dfJobs.loc[jobID, 'UTCEnd'].strftime("%Y-%m-%dT%H:%M:%SZ")

    if sum(dfJobs.index == jobID) > 1:
        lNodeList = list(dfJobs.loc[jobID, 'NodeList'])
    else:
        lNodeList = [dfJobs.loc[jobID, 'NodeList']]

    sNodeQuery = '|'.join(lNodeList)

    data = {
        'query': f'amperageProbeReading{{alias=~"{sNodeQuery}", amperageProbeLocationName="System Board Pwr Consumption"}}',
        'start': sStart,
        'end' : sEnd,
        'step': '30s'
    }

    try:
        response = requests.put(
        url, 
        data=data,
        proxies=proxies,
        headers=headers,
        timeout=10
    )  
    except requests.exceptions.ConnectionError as e:
        print('ConnecionError')
        return None
    except requests.exceptions.ReadTimeout as e:
        print('ReadRimeout')
        return None

    if (response.status_code != 200):
        print('Response status code was not 200.')
        print(f'The response was {response.status_code}')
        return None

    if len(response.json()['data']['result']) == 0:
        print('No data.')
        return None

    dNodePowers = {}
    dPowerData = {}
    lTicks = []
    lNodes = []

    for dNodeData in response.json()['data']['result']:
        sNode = dNodeData['metric']['alias']
        
        if sNode in lNodes:
            continue 

        lData = dNodeData['values']
        lNodes.append(sNode)
        dNodePowers[sNode] = lData

        for lDataPoint in lData:
            iTick = lDataPoint[0]
            iPower = lDataPoint[1]
            lTicks.append(iTick)
            dPowerData[(iTick, sNode)] = iPower
    
    lTicks.sort()
    setTicksOrdered = set(lTicks)

    dfJobPower = pd.DataFrame(
        index = setTicksOrdered,
        columns = lNodes
    )

    for tIndex in dPowerData.keys():
        dfJobPower.loc[tIndex[0], tIndex[1]] = dPowerData[tIndex]

    dfJobPower.index = pd.to_datetime(dfJobPower.index, unit='s', utc=True)
    dfJobPower['Date'] = dfJobPower.index.strftime('%Y-%m-%d')
    dfJobPower['Date'] = dfJobPower['Date'].str.cat((((dfJobPower.index).hour * 2) + ((dfJobPower.index).minute//30) + 1).astype(str), sep=" ")

    dfJobPower[dfJobPower.columns[:-1]] = dfJobPower[dfJobPower.columns[:-1]].apply(pd.to_numeric, axis=1)
    dfJobPower = dfJobPower.resample('30S', origin='start').interpolate()

    return dfJobPower

def getJobEnergy(dfPowerData):
    """
    Returns a DataFrame containing the energy consumed by each node the job runs on (in Wh) for each 30 minute time period of the Job's duration. 

    Parameters 
    ----------
    dfPowerData: pdDataFrame
        The DataFrame containin the Victoria Metrics power data for each node the job runs on for the job's running period. 
        This DataFrame is returned by the getJobPower() function.  

    Returns
    ----------
    dfNodeEnergies: pdDataFrame
        The DataFrame containing the energy consumed by each node (in Wh) for each 30 minute time period. 
    """

    dEnergies = {}

    for sNode in dfPowerData.columns[:-1]:
            dIntervalEnergies = {}

            lPeriodDFs = []

            for interval in dfPowerData['Date'].unique():                
                bIntervalMask = dfPowerData['Date'] == interval
                lPeriodDFs.append(dfPowerData[bIntervalMask])

            for dfIndex in range(len(lPeriodDFs)):
                interval = lPeriodDFs[dfIndex]['Date'].unique()[0]

                if dfIndex != 0:
                        dfIntervalPower = pd.concat([lPeriodDFs[dfIndex - 1].iloc[-1].to_frame().transpose(), lPeriodDFs[dfIndex]])
                else:
                        dfIntervalPower = lPeriodDFs[dfIndex]

                if dfIntervalPower[sNode].isnull().values.any():
                        dIntervalEnergies[interval] = None
                        continue 

                iJoules = np.trapz(dfIntervalPower[sNode].astype(int), dx=30)
                iWattHour = iJoules/3600

                dIntervalEnergies[interval] = iWattHour

            dEnergies[sNode] = dIntervalEnergies

    dfNodeEnergies = pd.DataFrame.from_dict(dEnergies)
    
    return dfNodeEnergies

def getCarbonIntensities(dfJobs):
    """ 
    Returns a DataFrame of the carbon intensities for each 30 minute time period in the interval.

    Checks whether the .csv file containing the carbon intensities for the given month, with the 
    format 'CarbonIntensities<Month>.csv' already exists. If it exists, returns a DataFrame, from 
    the .csv file, whose index is the time period, in the format 'YYYY-MM-DD PERIOD' where PERIOD 
    is the 30 minute time period of that date as an integer from 1-48. The DataFrame contains all 
    carbon intensities, in gCO2/kWh, for each 30 minute time period in the given interval. If the 
    .csv file does not exist, the DataFrame with the format above will be created and saved into a
    .csv file, before being returned.

    Parameters
    ----------
    dfJobs: pdDataFrame
        The DataFrame containing all of the job data for the given time period. 

    Returns
    ----------
    dfIntensities: pdDataFrame 
        A DataFrame containing the carbon intensity, in gCO2/kWh, for each 30 minute interval within
        the specified time range. 
    None: NoneType
        Returns 'None' if there is a problem accessing the carbon intensity API.
    """

    sMonth = dfJobs['UTCStart'].dt.month_name(locale='English').value_counts().index[0]
    sFileName = 'CarbonIntensities' + sMonth + '.csv'

    fCarbonData = open(sFileName, 'a+')

    fCarbonData.seek(0)
    bEmpty = len(fCarbonData.read()) == 0

    fCarbonData.close()

    if bEmpty:
        start = dfJobs.iloc[0]['UTCStart']
        end = dfJobs.iloc[-1]['UTCEnd']

        timeDeltaSeconds = (end-start).total_seconds()
        iChunks = int(np.ceil(timeDeltaSeconds/(30 * 86400)))

        lCarbonDFs = []

        for iCount in range(iChunks):
            start = start + pd.Timedelta((30 * iCount), 'd')
            if iCount < iChunks - 1:
                tempEnd = start + pd.Timedelta(30, 'd')
            else:
                tempEnd = end.date() + pd.Timedelta(24, 'h')
            
            sStart = start.strftime("%Y-%m-%dT%H:%MZ")
            sEnd = tempEnd.strftime("%Y-%m-%dT%H:%MZ")

            intensity = requests.get(f'https://api.carbonintensity.org.uk/intensity/{sStart}/{sEnd}')

            if (intensity.status_code == 400):
                print('Status code was 400.')
                print('There was a bad request.')
                return None
            elif (intensity.status_code == 500):
                print('Status code was 500.')
                print('There was an internal server error.')
                return None

            dfTempIntensities = pd.DataFrame(intensity.json()['data'])

            dfTempIntensities['from'] = pd.to_datetime(dfTempIntensities['from'])
            dfTempIntensities['date'] = dfTempIntensities['from'].dt.date
            dfTempIntensities['period'] = dfTempIntensities['date'].astype(str) + " " + ((dfTempIntensities['from'].dt.hour * 2) + (dfTempIntensities['from'].dt.minute//30) + 1).astype(str)

            dfTempIntensities.drop(columns=['from', 'to', 'date'], inplace=True)
            dfTempIntensities['intensity'] = pd.json_normalize(dfTempIntensities['intensity'])['actual']
            dfTempIntensities.set_index('period', inplace=True)

            lCarbonDFs.append(dfTempIntensities)

        dfIntensities = pd.concat(lCarbonDFs)

        dfIntensities.to_csv(sFileName)
        
        return dfIntensities

    dfIntensities = pd.read_csv(sFileName, parse_dates=[0], infer_datetime_format=True)
    dfIntensities.set_index('period', inplace=True)

    return dfIntensities

def getCarbonFootprint(jobID, df):
    """
    Returns a DataFrame containing the job's carbon footprint data.

    Returns a DataFrame containing the job's energy consumption, in Wh; 
    carbon footprint, in gCO2; and the distance driven by a medium sized
    diesel car, in km, that releases the same amount of carbon dioxide. 

    Parameters 
    ----------
    jobID: integer
        The integer job ID of the job whose carbon footprint is calculated.
    df: pdDataFrame
        The pandas DataFrame containing all the job data.

    Returns
    ----------
    dfCarbonData: pd.DataFrame
        The pd DataFrame containing the job's carbon data.  

    """

    dfJobPower = getJobPower(jobID, df)

    if isinstance(dfJobPower, pd.DataFrame) and dfJobPower.isnull().values.any():
        print(f'Missing Data for job: {jobID}')
        return pd.DataFrame([np.nan, np.nan]) 
    elif isinstance(dfJobPower, type(None)):
        print(f'Missing Data for job: {jobID}')
        return pd.DataFrame([np.nan, np.nan]) 
    
    
    dfJobEnergy = getJobEnergy(dfJobPower)

    if '2023-07-27 16' in dfJobEnergy.index:
        print(jobID)

    dfJobCarbonIntensities = getCarbonIntensities(df).loc[dfJobEnergy.index]

    dfJobEnergy.rename(columns={dfJobEnergy.columns[0] : 'Data'}, inplace=True)
    dfJobCarbonIntensities.rename(columns={dfJobCarbonIntensities.columns[0] : 'Data'}, inplace=True)

    iEnergyTotal = sum(dfJobEnergy[dfJobEnergy.columns[0]])

    dfJobCarbon = (dfJobEnergy * dfJobCarbonIntensities)/1000

    iCarbon = round(sum(dfJobCarbon['Data']))

    iDistance = iCarbon/171

    dfCarbonData = pd.DataFrame([iCarbon, iEnergyTotal, iDistance])

    return dfCarbonData

# We will now apply our functions above to calculate the carbon footprints of the jobs
dfSacct = dfToUTC(dfSacct)

lUserDFs = []

for user in dfSacct['User'].unique():
    bUserMask = dfSacct['User'] == user
    lUserDFs.append(dfSacct[bUserMask])

def findCarbonEnergy(df):
    df[['CarbonFootprint(gCO2)', 'TotalEnergy', 'EquivalentDistance(km)']] = df.apply(lambda row : getCarbonFootprint(row.name, df)[0], axis=1)

    return df

lUserDFs = [lUserDFs[-1]]

lCarbonDFs = Parallel(n_jobs=8)(delayed(findCarbonEnergy)(df) for df in lUserDFs)

lFinalDF = pd.concat(lCarbonDFs)

lFinalDF.to_csv(sys.argv[3])