Installation
------------

```bash
conda env create -f environment.yml
conda activate kalman
pip install -e .
```


Data download
-------------


```bash
cd data
datalad clone https://github.com/OpenNeuroDatasets/ds004148.git
cd ds004148
datalad get sub-01_ses-session2_task-eyesopen_eeg.eeg sub-01_ses-session2_task-eyesopen_eeg.vhdr  sub-01_ses-session2_task-eyesopen_eeg.vmrk
```
