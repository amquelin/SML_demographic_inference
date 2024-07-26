## SML for demographic inference

This repository is based on the paper : Simulation-based supervised machine learning for inferring demographic history from genomic data" and contains :

- a directory [Simulations](Simulations/) containing
  - the code to produce simulated genomic data under an isolation with migration models.
  - the code to compute summary statistics on these simulations.

- a directory [Demo_inference](Demo_inference/) containing
  - the code to develop parameter inference from summary statistics.
  - the hyperparameter files of the ML models.
  - a notebook to show results of trained models for an example of a demographic model.
 

### Pipeline summary

- `ms_simulations.py` produces the summary statistics computed on simulated data from settings defined in `default_settings.py`.
- The code for summary statistics computation is `summary_statistics.py`.
- Once computing, the summary statistics are used to train inference models with `modelling.py` (see example.ipynb).
