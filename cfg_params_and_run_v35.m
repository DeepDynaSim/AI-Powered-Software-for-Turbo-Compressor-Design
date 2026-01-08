addpath(genpath("src"));

cfg = struct();
cfg.inputMat  = "input.mat";
cfg.outputMat = "output.mat";
cfg.extInputMat  = "input.mat";   % if you have ext_input.mat, set here
cfg.extOutputMat = "output.mat";  % if you have ext_output.mat, set here
cfg.speedCol = 1;
cfg.in23Col  = 23;
cfg.gasName  = "air";

cfg.modelOut  = "turbo_model_v35.mat";
cfg.reportOut = "turbo_external_report_v35.mat";
cfg.etaTuneOnExternal = true;   % tune w0 using external (as in your v33 gain)

run_v35_train_and_external(cfg);
