# G2DA: GMM-Based Two-Stage DEA-DA for AML Prediction

This repository provides the simulation and analysis code accompanying the manuscript **“A Fusion Framework Integrating GMM Clustering with a Two-Stage DEA-DA Model for AML Prediction.”**

G2DA is designed for binary prediction under extreme class imbalance and latent population heterogeneity. The framework combines:

1. **Gaussian mixture modelling (GMM)** to identify latent subgroups;
2. **an adaptive class-weighting parameter, beta**, to redistribute classification errors between the majority and minority classes;
3. **direction-constrained DEA-DA optimization**, which incorporates prior knowledge about the expected directions of clinical indicators; and
4. **a second-stage local update**, which refines samples located in the uncertainty region around the Stage-I decision boundary.

In the accompanying study, `target = 1` denotes AML and `target = 0` denotes non-AML. Because false-negative AML predictions are clinically important, the evaluation emphasizes recall and F1-score in addition to AUC, precision, and accuracy.

## Repository contents

| File | Purpose |
| --- | --- |
| `code/data_simulated_rep_9groups.py` | Trains a CTGAN model and generates Monte Carlo datasets with controlled sample sizes and positive-class proportions. |
| `code/GMM_Clustering.m` | Fits GMMs on the training data, selects the number of components using BIC, and assigns training and test observations to latent clusters. |
| `code/jiaoben8_1_7_simu0708_beta9_geometric_visualization_mode.m` | Main G2DA simulation script, including model selection, evaluation, result aggregation, and visualization-only mode. |
| `code/jiaoben8_1_7_simu0629.m` | Earlier main G2DA simulation implementation retained for reference. |
| `code/model8_1_7.m` | Stage-I DEA-DA optimization model. |
| `code/model8_1_8.m` | Stage-II local feature-weight update for observations in the uncertainty region. |
| `code/lr_baseline_CTGAN_0713.m` | Logistic-regression benchmark. |
| `code/svm_baseline_CTGAN_0629.m` | Support-vector-machine benchmark. |
| `code/knn_baseline_CTGAN_0629_v2.m` | k-nearest-neighbors benchmark. |
| `code/rf_baseline_CTGAN_0629.m` | Random-forest benchmark using MATLAB `TreeBagger`. |
| `code/xgb_baseline_CTGAN_0629.m` | XGBoost benchmark through MATLAB's Python interface. |
| `code/lgbm_baseline_CTGAN_0629.m` | LightGBM benchmark through MATLAB's Python interface. |

## Simulation design

The Python generator is configured to train one CTGAN model and generate 100 independently shuffled datasets for each combination of:

- sample size: `N = 3000, 5000, 8000`;
- AML-positive proportion: `pi = 0.01, 0.03, 0.05`.

This produces nine simulation settings and 900 synthetic datasets in total. Files are named:

```text
synthetic_CTGAN_001.csv
synthetic_CTGAN_002.csv
...
synthetic_CTGAN_100.csv
```

The manuscript uses two complementary comparisons:

- a sample-size experiment with `pi = 0.03` and `N = 3000, 5000, 8000`;
- a class-imbalance sensitivity experiment with `N = 5000` and `pi = 0.01, 0.03, 0.05`.

Each dataset is divided by stratified random sampling into 70% training data and 30% test data. The supplied MATLAB scripts use random seed `9` and three-fold cross-validation by default. In the main G2DA script, candidate GMM sizes range from 3 to 6 and the final number of clusters is selected using the Bayesian information criterion (BIC), subject to the minimum positive-sample requirement.

## Data format

Input CSV files must have the following structure:

- the **first column** is the binary outcome `target`;
- `target = 1` represents AML and `target = 0` represents non-AML;
- all remaining columns are model predictors;
- predictor columns supplied to the MATLAB models must be numeric and consistently ordered across files.

The CTGAN generator currently also marks `Gen` as categorical. If the input dataset does not contain a column named `Gen`, remove or modify the corresponding `metadata.update_column(...)` statement.

## Requirements

### Python

- Python 3
- `pandas`
- `sdv`
- `sdmetrics`
- `numpy`, `xgboost`, and `lightgbm` for the MATLAB XGBoost and LightGBM benchmarks

Install the Python dependencies with:

```bash
python -m pip install pandas sdv sdmetrics numpy xgboost lightgbm
```

### MATLAB

- MATLAB with the Statistics and Machine Learning Toolbox
- Optimization Toolbox
- a Python environment visible to MATLAB for the XGBoost and LightGBM benchmarks

The MATLAB code uses functions including `fitgmdist`, `fitcsvm`, `fitcknn`, `fitglm`, `TreeBagger`, `tiedrank`, `pca`, `linprog`, `quadprog`, and `intlinprog`.

To inspect or configure the Python environment used by MATLAB:

```matlab
pyenv
```

If necessary, select the required Python executable before running the XGBoost or LightGBM scripts:

```matlab
pyenv(Version="C:\path\to\python.exe")
```

Restart MATLAB after changing the Python environment when required by your MATLAB release.

## Reproducing the simulation study

### 1. Prepare the project

Clone the repository and open its directory:

```bash
git clone https://github.com/yyc-333/gmm-adaptive-da-dea-aml-simu.git
cd gmm-adaptive-da-dea-aml-simu
```

Keep all MATLAB function files on the MATLAB path. For example:

```matlab
addpath(genpath(pwd));
```

### 2. Generate the synthetic datasets

Open `code/data_simulated_rep_9groups.py` and replace the two local Windows paths:

```python
data = pd.read_csv(r"PATH_TO_PREPROCESSED_AML_DATA.csv")
base_output_dir = r"PATH_TO_SIMULATION_OUTPUT"
```

Then run:

```bash
python code/data_simulated_rep_9groups.py
```

The source cohort is used only to fit CTGAN and evaluate synthetic-data quality. It is **not included in this repository** because it contains protected clinical information. Users who already have the released synthetic CSV files can skip this step.

### 3. Run G2DA

Open `code/jiaoben8_1_7_simu0708_beta9_geometric_visualization_mode.m` and update:

```matlab
dataFolder = 'PATH_TO_ONE_SIMULATION_SETTING';
resultRootFolder = 'PATH_TO_RESULTS';
```

For a complete model run, set:

```matlab
runVisualizationOnly = false;
```

Run the script from MATLAB. It calls the following files, which must remain accessible on the MATLAB path:

```text
GMM_Clustering.m
model8_1_7.m
model8_1_8.m
```

Repeat the analysis after changing `dataFolder` to each simulation setting required for the corresponding experiment.

### 4. Recreate figures without refitting the model

After a complete run has produced the required result CSV files, set:

```matlab
runVisualizationOnly = true;
```

Re-run `code/jiaoben8_1_7_simu_beta9.m`. In this mode, the script reads the existing `synthetic_CTGAN_*_global_micro_metrics.csv` files and regenerates the summary tables and figures without repeating model fitting.

### 5. Run the benchmark models

For each benchmark script, update `dataFolder`, `resultRootFolder`, and `resultFolder`, then run the desired file in MATLAB:

```text
code/lr_baseline_CTGAN.m
code/svm_baseline_CTGAN.m
code/knn_baseline_CTGAN.m
code/rf_baseline_CTGAN.m
code/xgb_baseline_CTGAN.m
code/lgbm_baseline_CTGAN.m
```

Use the same synthetic datasets and data partitions when comparing methods. The benchmark scripts perform cluster-level modelling and then aggregate predictions using micro-level confusion counts and pooled prediction scores.

## Main analytical settings

The supplied main script uses the following defaults:

| Setting | Default |
| --- | --- |
| Train/test split | 70% / 30%, stratified by outcome |
| Random seed | 9 |
| Cross-validation | 3 folds, adaptively reduced when necessary |
| Feature-group correlation threshold | 0.90 |
| Candidate GMM components | 3-6 |
| Minimum AML-positive training observations per cluster | 12 during GMM construction |
| Number of beta positions | 9 |
| Eta in the supplied simulation script | 1 |

Feature grouping, GMM fitting, cross-validation, and parameter selection are performed using training data. Test data are reserved for independent performance evaluation.

## Outputs

Depending on the model and selected options, the scripts produce:

- cluster-specific training and test assignments;
- cross-validation, training, and test metrics;
- aggregated confusion counts (`TP`, `FP`, `FN`, and `TN`);
- AUC, precision, recall, F1-score, and accuracy summaries;
- Stage-I and Stage-II comparisons;
- beta-sensitivity summaries and figures;
- feature-weight and cluster-level visualization files.

Overall precision, recall, F1-score, and accuracy are calculated from confusion counts aggregated across latent clusters. Overall AUC is calculated from pooled individual prediction scores. In G2DA, a smaller raw discriminant score indicates greater support for the AML-positive class; the code reverses the score direction where needed so that reported AUC consistently treats AML as the positive class.

## Expected simulation results

The accompanying manuscript reports that, at `pi = 0.03`, G2DA achieved mean test recall values of approximately 0.75, 0.80, and 0.82 for `N = 3000`, `N = 5000`, and `N = 8000`, respectively. The corresponding mean test F1-scores were approximately 0.75, 0.79, and 0.81. Exact values may vary if the source data, package versions, random seeds, numerical solver, or code settings are changed.

## Data availability and privacy

This repository is intended to distribute code and simulated data only. The retrospective clinical cohort used to train the generative model is not publicly distributed because of privacy and institutional data-governance requirements. Do not commit identifiable patient data, local credentials, API keys, or other sensitive information to this repository.

Synthetic data should not automatically be assumed to be free of disclosure risk. Before public release, the generated files should be reviewed under the applicable ethics approval, institutional policy, and data-sharing agreement.

## Citation

If you use this code, please cite the accompanying manuscript:

> *A Fusion Framework Integrating GMM Clustering with a Two-Stage DEA-DA Model for AML Prediction.*

Full bibliographic information will be added after publication.

## License

No license is granted unless a `LICENSE` file is added to the repository. Before public release, select a license consistent with the manuscript, institutional requirements, and any third-party software dependencies.

## Disclaimer

This software is provided for research and reproducibility purposes only. It is not a medical device and must not be used as a substitute for professional diagnosis, clinical judgment, or validated laboratory procedures.
