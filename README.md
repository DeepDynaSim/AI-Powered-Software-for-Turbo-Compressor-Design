# Physics-Guided AI for Turbocompressor Performance

**DeepDynaSim Engineering & Consulting | MATLAB research workflow | turbo-pgml v35**

Predict turbocompressor performance with a surrogate that combines regression models, thermodynamic relationships and residual correction. This repository includes training, inference, evaluation data, a saved model and an interactive MATLAB reporting dashboard.

[Quick start](#quick-start) · [Method](#method-overview) · [Results](#results-and-reporting) · [Technical report](Report.pdf) · [Presentation](Generative_Turbo_Design_Presentation.pdf)

## What you can explore

- Five outputs: efficiency **ETA**, total pressure ratio **TPR**, entropy-related **ENT**, **ALPHA2** and **VTH2**.
- Physics-guided entropy coupling, residual learning and efficiency blending.
- Internal cross-validation, holdout metrics and evaluation on a supplied external set.
- Parity/residual plots, speed-bin diagnostics, CSV exports and PDF reporting.

This is a research software workflow for performance prediction. Its scope is the implemented surrogate and evaluation tools; application to a new compressor or operating domain requires validation with suitable data.

## Repository map

All MATLAB files currently live in the repository root.

| File | Purpose |
| --- | --- |
| [run_v35_train_and_external.m](run_v35_train_and_external.m) | Train and evaluate a configured experiment |
| [train_turbocompressor_model_v35.m](train_turbocompressor_model_v35.m) | Fit the v35 surrogate |
| [predict_turbocompressor_model_v35.m](predict_turbocompressor_model_v35.m) | Predict the five outputs |
| [sanity_check_ent_dsfactor_v35.m](sanity_check_ent_dsfactor_v35.m) | Inspect entropy/physics consistency |
| [turbo_results_dashboard_ui_v35.m](turbo_results_dashboard_ui_v35.m) | Visualize and export evaluation results |
| [cfg_params_and_run_v35.m](cfg_params_and_run_v35.m) | Existing configuration example; review its data and tuning settings |
| `input.mat`, `output.mat`, `ext_input.mat`, `ext_output.mat` | Included input/output matrices |
| `turbo_model_v35.mat`, `turbo_external_report_v35.mat` | Saved model and report |

## Quick start

Use MATLAB with Statistics and Machine Learning Toolbox for the regression models. Open MATLAB in the cloned repository directory. The dashboard uses MATLAB desktop UI components.

```sh
git clone https://github.com/DeepDynaSim/AI-Powered-Software-for-Turbo-Compressor-Design.git
cd AI-Powered-Software-for-Turbo-Compressor-Design
```

### Inspect a saved model

```matlab
addpath(pwd);
S = load("input.mat");
X = double(load_matrix_any(S, "input"));
Yhat = predict_turbocompressor_model_v35("turbo_model_v35.mat", X);
% Columns: ETA, TPR, ENT, ALPHA2, VTH2
```

### Train and evaluate

Check the column meanings and working gas before running. Input and output rows must correspond; output columns are ordered as above. This example uses the included external filenames and explicitly disables external-label tuning.

```matlab
addpath(pwd);
cfg = struct();
cfg.inputMat = "input.mat";
cfg.outputMat = "output.mat";
cfg.extInputMat = "ext_input.mat";
cfg.extOutputMat = "ext_output.mat";
cfg.speedCol = 1;
cfg.in23Col = 23;
cfg.gasName = "air";  % or "hydrogen", as appropriate for your data
cfg.modelOut = "turbo_model_v35.mat";
cfg.reportOut = "turbo_external_report_v35.mat";
cfg.etaTuneOnExternal = false;
run_v35_train_and_external(cfg);
```

Training writes the model/report paths above; use different filenames to retain the bundled artifacts. The existing configuration script uses the internal filenames for external evaluation and enables external-label tuning. Reusing training data is a demonstration, not independent validation. Any dataset used for tuning must be separated from the final untouched test set. Separate filenames alone do not establish independence.

### Open the dashboard

```matlab
turbo_results_dashboard_ui_v35( ...
    "modelPath", "turbo_model_v35.mat", ...
    "extInput", "ext_input.mat", ...
    "extOutput", "ext_output.mat", ...
    "predictFcn", "predict_turbocompressor_model_v35", ...
    "speedIdx", 1);
```

## Method overview

1. **TPR:** select candidate regressors/transforms using cross-validation.
2. **ETA:** fit engineered features in logit space and apply polynomial bias correction.
3. **ENT:** couple efficiency, pressure ratio and the configured input column through the thermodynamic `dsfactor` relationship; learn a residual correction.
4. **Efficiency blending:** combine the data-driven estimate with the physics-derived estimate using the v35 blending/gating model.
5. **ALPHA2 and VTH2:** fit separate candidate regressors.

## Results and reporting

The dashboard reports RMSE, MAE and R², input-shift summaries, parity plots and residuals. Exports include `metrics.csv`, `shift.csv`, `eta_bins.csv`, `rows_preview.csv`, output PNGs and a multipage PDF. Inspect [ETA](output_ETA.png), [TPR](output_TPR.png) and [ENT](output_ENT.png) plots already in the repository.

Label training, internal holdout, tuning and final external results separately when reporting an experiment. Report the dataset provenance, split, MATLAB/toolbox versions, gas choice and column definitions with the metrics.

## Troubleshooting

- **Missing function:** use the repository root as MATLAB current folder and run `addpath(pwd)`. There is no `src/` directory in this checkout.
- **Unexpected entropy results:** check `in23Col`, `gasName` and the third output column.
- **Regression/toolbox errors:** check Statistics and Machine Learning Toolbox availability.
- **Different MAT field names:** inspect the matrices and use `load_matrix_any` or adjust the loading step.

## Maintainers, collaboration and license

Maintained by **Caglar Uyulan, DeepDynaSim Engineering & Consulting Ltd. Co.** Contributor: **Sercan Acarer, SLC Fluidics Ltd.**

DeepDynaSim is an engineering R&D and consulting company in İzmir, Türkiye, working on physics-guided AI, simulation and control. For engineering collaboration: [support@deepdynasim.com](mailto:support@deepdynasim.com). For reproducible bugs or technical questions, [open an issue](https://github.com/DeepDynaSim/AI-Powered-Software-for-Turbo-Compressor-Design/issues) with the MATLAB version, toolbox list, configuration and a minimal example.

Code is distributed under the [MIT license](LICENSE). When using this work academically, reference the repository commit or version and the applicable literature.

Related work: [Satellite thermal preview control](https://github.com/DeepDynaSim/satellite-thermal-preview-control) · [Deep Lyapunov functions](https://github.com/DeepDynaSim/DeepLyapunovFunction) · [Flexible-link manipulators](https://github.com/DeepDynaSim/Flexible-Link-Manipulators-Modelling-and-Control)

## References (selected)


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
