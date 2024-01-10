# A Study of Pre-processing Fairness Intervention Methods for Ranking People
## Requirements
Set up the conda env with python 3.8.2 from the env.yml file to run experiments with CIFRank, LFR and iFair.
```
conda env create -f env.yml
conda activate fairness_env 
```

Set up the conda env with python 3.7 from the env_fair.yml file to run experiments with FA*IR.
```
conda env create -f env_fair.yml
conda activate fair 
```

## Run
To run the experiments run the following command:
```
python run.py --config_path <config_file_path>
```
The paths of the config files can be found in the configs folder. Before running the command download the required dataset and update the path to the dataset in the config file.

## References
[1]
Rus, Clara, Maarten de Rijke, and Andrew Yates. "A Study of Pre-processing Fairness Intervention
Methods for Ranking People." (2024).

[2]
Ke Yang, Joshua R. Loftus, and Julia Stoyanovich. 2021. Causal intersectionality and fair ranking. In Symposium on Foundations of Responsible
Computing (FORC). 

[3]
Preethi Lahoti, Krishna P Gummadi, and Gerhard Weikum. 2019. ifair: Learning individually fair data representations for algorithmic decision
making. In 2019 IEEE 35th International Conference on Data Engineering (ICDE). IEEE, 1334–1345.

[4]
Zemel, Rich, et al. "Learning fair representations." International conference on machine learning. PMLR, 2013.

[5]
Meike Zehlike, Francesco Bonchi, Carlos Castillo, Sara Hajian, Mohamed Megahed, and Ricardo Baeza-Yates. 2017. Fa* ir: A fair top-k ranking
algorithm. In Proceedings of the 2017 ACM on Conference on Information and Knowledge Management. ACM, 1569–1578.
