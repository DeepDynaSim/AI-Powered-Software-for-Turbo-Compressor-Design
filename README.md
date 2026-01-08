# AI-Powered-Software-for-Turbo-Compressor-Design
Physics-guided surrogate modeling for turbomachinery: entropy/efficiency-consistent learning, residual correction, and robust external validation.
# turbo-pgml (v35) — Physics-Guided Turbocompressor Surrogate

Physics-guided machine learning pipeline for turbocompressor performance prediction with a focus on **efficiency (ETA, η)** and **entropy (ENT)** generalization.  
Includes **training**, **inference**, **external validation**, and a **MATLAB dashboard UI** for reporting.

## What this project does

This repository provides a complete MATLAB workflow to train and deploy a surrogate model that predicts:

- **ETA (η)** — compressor efficiency  
- **TPR** — total pressure ratio  
- **ENT** — entropy-related output (physics-consistent via dsfactor coupling)  
- **ALPHA2**, **VTH2** — additional turbomachinery outputs

The v35 pipeline is **physics-guided**: it blends data-driven learning with a thermodynamics-based relationship (via `dsfactor`) and uses a **gating / blending** mechanism to improve out-of-domain (external) performance for ETA.

## Why it’s useful

Classical CFD-based approaches can be expensive and may generalize poorly outside the training domain.  
Purely data-driven models can overfit or violate physics constraints.

This project demonstrates a pragmatic “PGML” (Physics-Guided ML) approach:

- Keeps the model **fast** (surrogate)
- Improves **external** ETA/ENT robustness
- Provides **reproducible reporting** (CSV + PDF + plots) via UI

## Repository structure

Recommended layout:
├─ src/
│ ├─ train_turbocompressor_model_v35.m
│ ├─ predict_turbocompressor_model_v35.m
│ ├─ run_v35_train_and_external.m
│ ├─ sanity_check_ent_dsfactor_v35.m
│ └─ turbo_results_dashboard_ui_v35.m
├─ cfg_params_and_run_v35.m
├─ input.mat
├─ output.mat
├─ (optional) ext_input.mat
├─ (optional) ext_output.mat
└─ README.md

> **MAT format expectation**
- `input.mat` contains a numeric matrix named `input` (or another numeric matrix field).
- `output.mat` contains a numeric matrix named `output` with at least `N x 5`.

## Quick start

### 1) Requirements
- MATLAB (Statistics and Machine Learning Toolbox recommended for GPR / ensembles)
- Your dataset files:
  - `input.mat`
  - `output.mat`

### 2) Add paths
```matlab
addpath(genpath("src"));

Train + evaluate (internal + external)

Create or edit cfg_params_and_run_v35.m:

addpath(genpath("src"));

cfg = struct();
cfg.inputMat  = "input.mat";
cfg.outputMat = "output.mat";

% External evaluation (can be same as internal if you don't have a separate set)
cfg.extInputMat  = "input.mat";
cfg.extOutputMat = "output.mat";

% Column indices
cfg.speedCol = 1;     % speed column index in X
cfg.in23Col  = 23;    % input23 column index in X (used in dsfactor)
cfg.gasName  = "air"; % "air" or "hydrogen"

% Outputs
cfg.modelOut  = "turbo_model_v35.mat";
cfg.reportOut = "turbo_external_report_v35.mat";

% Optional tuning
cfg.etaTuneOnExternal = true;

run_v35_train_and_external(cfg);


Run:

cfg_params_and_run_v35

Predict
X = load("input.mat");         % or your new matrix
Xmat = X.input;                % adapt if your field name differs

Yhat = predict_turbocompressor_model_v35("turbo_model_v35.mat", Xmat);

Dashboard UI (evaluation + export)
turbo_results_dashboard_ui_v35( ...
  "modelPath","turbo_model_v35.mat", ...
  "extInput","input.mat", ...
  "extOutput","output.mat", ...
  "predictFcn","predict_turbocompressor_model_v35", ...
  "speedIdx",1);


The UI can export:

metrics.csv, shift.csv, eta_bins.csv

per-output PNG figures

a multi-page PDF report

Method overview (v35)

High-level stages:

TPR model
K-fold CV selection among candidate regressors/transforms.

ETA base model (data-driven)

Feature subset + engineered features (speed, log(speed), tpr, log(tpr), etc.)

Trained in logit(ETA) space

Polynomial bias correction (0–2 degree)

ENT physics coupling via dsfactor

Uses thermodynamic mapping:

ds = dsfactor(eta, tpr, input23, consts)

ENT is approximated as ENT ≈ a*ds + b (affine fit from data)

Learns a residual correction in ds-space (DS_CORR)

ETA physics inversion + blending (ETA-CYCLE / ETA-GATE)

Inverts dsfactor to get eta_phys

Blends eta_data and eta_phys using a tuned global w0

Applies a gated fine-tune around w0 with safeguards to prevent collapse

ALPHA2 & VTH2
CV-selected regressors for remaining outputs.

Results reporting

Typical reporting outputs:

Internal TrainVal OOF metrics (robust internal estimate)

Internal Holdout Test metrics (sanity check)

External validation metrics (the key KPI)

For your project report, prioritize:

External R²/RMSE for ETA and ENT

ETA bin-by-speed error analysis (eta_bins.csv)

Residual plots and parity plots (exported via UI)

Troubleshooting

“ENT/dsfactor sanity looks wrong”
Check:

cfg.in23Col points to the correct input23 column

output.mat uses column 3 for ENT (i.e., Y(:,3) is ENT)

gasName is correct (air vs hydrogen)

Toolbox errors (GPR / ensembles)
Ensure Statistics and Machine Learning Toolbox is installed.

MAT field names differ
The loader searches common alternatives, but best practice is:

input.mat contains input

output.mat contains output

Where to get help

Open an issue describing:

MATLAB version

Toolbox availability

cfg settings

Console logs (train + UI)

Sample row/column descriptions (especially speed and input23)

Maintainers

Maintained by: Caglar Uyulan (DeepDynaSim Engineering & Consulting Ltd. Co.)

Contributors: Sercan Acarer (SLC Fluidics Ltd.)

Contributing

Contributions are welcome. Please:

Open an issue first (bug/feature request)

Use clear commit messages

Add minimal reproducible examples for changes

(Optional) Add:

docs/CONTRIBUTING.md

CODE_OF_CONDUCT.md

LICENSE

CITATION.cff

Citation

If you use this repository in academic work, please cite the relevant literature below and reference this repository version/tag.

References (selected)

Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S., & Yang, L. (2021). Physics-informed machine learning. Nature Reviews Physics, 3, 422–440. https://doi.org/10.1038/s42254-021-00314-5

Karpatne, A., Atluri, G., Faghmous, J. H., Steinbach, M., Banerjee, A., Ganguly, A., … Kumar, V. (2017). Theory-guided data science: A new paradigm for scientific discovery from data. IEEE TKDE, 29(10), 2318–2331. https://doi.org/10.1109/TKDE.2017.2720198

Li, W., Montomoli, F., & Sharma, R. (2024). An adaptive physics-informed neural network methodology for modelling industrial gas turbines. AIAA Journal. https://doi.org/10.2514/1.J063562

Li, Z., Montomoli, F., & Sharma, S. (2023). Investigation of compressor cascade flow based on physics-informed neural networks. arXiv. https://doi.org/10.48550/arXiv.2308.04501

McClenny, L., & Braga-Neto, U. (2023). Self-adaptive physics-informed neural networks using a soft attention mechanism. Journal of Computational Physics, 474, 111722. https://doi.org/10.1016/j.jcp.2022.111722

Pawar, S., San, O., Aksoylu, B., Rasheed, A., & Kvamsdal, T. (2021). Physics guided machine learning using simplified theories. Physics of Fluids, 33(1), 011701. https://doi.org/10.1063/5.0038929

Pawar, S., San, O., Vedula, P., Rasheed, A., & Kvamsdal, T. (2022). Multi-fidelity information fusion with concatenated neural networks. Scientific Reports, 12, 5900. https://doi.org/10.1038/s41598-022-09938-8

Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear PDEs. Journal of Computational Physics, 378, 686–707. https://doi.org/10.1016/j.jcp.2018.10.045

Rathnakumar, A. S., et al. (2024). Bayesian entropy neural networks. arXiv. https://doi.org/10.48550/arXiv.2407.01015

Sunderland, E., et al. (2022). Multi-fidelity regression using artificial neural networks: Efficient approximation of parameterized simulations. Computer Methods in Applied Mechanics and Engineering, 389, 114378. https://doi.org/10.1016/j.cma.2021.114378

Willard, J., Jia, X., Xu, S., Steinbach, M., & Kumar, V. (2022). Integrating scientific knowledge with machine learning for engineering and environmental systems. ACM Computing Surveys. https://doi.org/10.1145/3514228

<img width="2048" height="2048" alt="InfoGraphic_TurboAI" src="https://github.com/user-attachments/assets/fce8ddcb-02b4-4b56-a385-8631c1111126" />
<img width="1536" height="1024" alt="Design CFD-Geometry" src="https://github.com/user-attachments/assets/e7afcf49-e703-450c-99af-9ac8c619021b" />
<img width="2151" height="1624" alt="output_ETA" src="https://github.com/user-attachments/assets/06f59dc0-6dd7-42c0-ba57-2adbb464cbc2" />
<img width="2143" height="1624" alt="output_ENT" src="https://github.com/user-attachments/assets/a65664ad-c6dc-4213-b383-8f818bfabb48" />
<img width="2127" height="1624" alt="output_ALPHA2" src="https://github.com/user-attachments/assets/02b716ec-183a-4fa7-93f1-1cb97978138a" />
<img width="2143" height="1624" alt="output_TPR" src="https://github.com/user-attachments/assets/f3c83db4-7487-4fe3-8759-30e817b58e2e" />
<img width="2151" height="1624" alt="output_VTH2" src="https://github.com/user-attachments/assets/fbda3632-c928-4916-bb48-3e569c7fa873" />
<img width="2160" height="1630" alt="eta_focus" src="https://github.com/user-attachments/assets/44a933b9-dd4e-429f-8638-13c5471fa448" />


