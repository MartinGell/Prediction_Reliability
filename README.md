# Brain-behaviour Predictions

This repository holds the scripts for prediction of behavioural and simualted behavioural data for the Burden of Reliability project (doi:...). Below is the necessary information to calculate predictions. While some pipelines have been tested to work out of the box, please note that this is not supposed to be a toolbox and is thus not polished to be one or maintained as one. That said, I am more than happy to answer any questions regarding the code and running it @ martygell@gmail.com

<br />

The main prediction script `prediction_nestedCV.py` is written to be ran in parallel on a computational cluster. Hence it can run in the command line and requires 4 arguments (if using submit files, these need to be defined in the respective submit file):

1. Functional connectivity file = `FC_file`
2. File with behavioural data = `beh_file`
3. Behaviour to predict = `beh`
4. Pipeline/algorithm (see defined pipelines in `code/func/models`) = `pipe`

<br />

Example files to run prediction script can be found in the directory. Example functional connectivity file in `/input/` (see below for more info on FC file) and example behavioural data file in `/text_files/`. Example to run in commandline:
```
$ python3 prediction_nestedCV.py Example_Schaefer400x17_hcpaging_2.jay HCP_A_age_example.csv interview_age ridgeCV_zscore
```

Other parameters (e.g. cross-validation, confound removal) have to be changed within the script itself.

<br />

### For replication of results
- Predictions using ridge regression were run with `pipe = ridgeCV_zscore` and with `pipe = ridgeCV_zscore_confound_removal_wcategorical` for analyses with confound regression
- Predictions using SVR with heuristic for C parameter were run with `pipe = svr_heuristic_zscore` and with `pipe = svr_heuristic_zscore_confound_removal_wcategorical` for analyses with confound regression
- For pipelines with featurewise confound regression note that inside `prediction_nestedCV.py` `remove_confounds` has to be set to `True` and desired confounds to regress selected after line 45.

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
`conda env create -f (cloned_dir)/reqs/env_setup.yml`

Check env was installed correctly:  
`conda info --envs`

There should now be a ((miniconda_dir)/envs/HCP_env) visible

To activate env:  
`conda activate HCP_env`

References: https://medium.com/swlh/setting-up-a-conda-environment-in-less-than-5-minutes-e64d8fc338e4

<br />

## 2. Input connectivity file
One FC file, shaped subject*connection(i.e. features) placed in the `/input/` folder. Features come from the upper (or lower) triangle of an FC matrix. There is an example file in the folder called 'Example_Schaefer400x17_hcpaging_2' which has 2 subjects and 78k (random) features.

For the file format I use .jay file format as it is faster to read but .csv also works, but few lines need to be changed (see below for details). To compile a .jay file you can use:

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
One csv file with named columns that contains target behaviour. Example file in the repository in `/text_files/`
