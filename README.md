# Prediction of simulated data

This repository has been created for prediction of behavioural and simualted behavioural data on Juseless. It has been optimised for faster computation times given that many predictions of simualted data need to be ran.

Main prediction script is `prediction_nestedCV.py`. to run it use the submit files. The submit file contains the most important options for prediction:
1. Pipeline/algorithm (see defined pipelines in code/func/models) = `pipe`
2. FC file = `dat`
3. Behaviour to predict = `beh`
4. File with behavioural data = `beh_f`


<br />


## Requirements:
1. Virtual environment
2. Input connectivity file
3. Input behavioural file

<br />

## 1. Environment
For this script to work a number of python modules are required. The easiest way to get these is using miniconda.

<br />

### Miniconda
In 'reqs' folder use the env_setup.yml to create the environemnt which will be called 'HCP_env':  
`conda env create -f (cloned_dir)/Preprocess_HCP/reqs/env_setup.yml`

Check env was installed correctly:  
`conda info --envs`

There should now be a ((miniconda_dir)/envs/HCP_env) visible

To activate env:  
`conda activate HCP_env`

References: https://medium.com/swlh/setting-up-a-conda-environment-in-less-than-5-minutes-e64d8fc338e4

<br />

## 2. Input connectivity file
One FC file, shaped subject*connection(i.e. features) placed in the `/input` folder. Features come from the upper (or lower) triangle of an FC matrix. There is an example file in the folder called 'Schaefer400x17_WM+CSF+GS_hcpaging_695.jay' which has 1 subjects and 78k (random) features.

For the file format I use .jay file format as it is faster to read but. Csv also works, but few lines need to be changed (see below for details). To compile a .jay file you can use:

```
import datatable as dt
import pandas as pd

pd.read_csv('FC.csv')
DT = dt.Frame(FC)
DT.to_jay('FC.jay')
```

Note that his may take up to a couple hours for thousands of subjects.

To use a .csv instead of .jay, swap the comments out in these two lines in `prediction_nestedCV.py`: 
```
#FCs_all = pd.read_csv(path2FC)
FCs_all = dt.fread(path2FC)
```

<br />

## 3. Input behavioural file
One csv file with named collumns that contains target behaviour. Example file in the repository
