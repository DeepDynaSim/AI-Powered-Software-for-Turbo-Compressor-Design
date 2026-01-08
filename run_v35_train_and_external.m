function out = run_v35_train_and_external(cfg)
%RUN_V35_TRAIN_AND_EXTERNAL Train turbo_v35 model and evaluate on external set.
%
% Required cfg fields:
%   cfg.inputMat   (string/char) path to input.mat
%   cfg.outputMat  (string/char) path to output.mat
%   cfg.speedCol   (scalar)      column index for SPD in X
%   cfg.in23Col    (scalar)      column index for input23 in X
%   cfg.gasName    (string/char) "air" or "hydrogen"
%
% Optional:
%   cfg.extInputMat, cfg.extOutputMat (if omitted, uses inputMat/outputMat)
%   cfg.modelOut    (default "turbo_model_v35.mat")
%   cfg.reportOut   (default "turbo_external_report_v35.mat")
%   cfg.K           (default 5)
%   cfg.testFrac    (default 0.15)
%   cfg.etaTuneOnExternal (default true)
%   cfg.rngSeed     (default 42)

assert(isfield(cfg,'inputMat')  && ~isempty(cfg.inputMat),  'cfg.inputMat required');
assert(isfield(cfg,'outputMat') && ~isempty(cfg.outputMat), 'cfg.outputMat required');
assert(isfield(cfg,'speedCol')  && isnumeric(cfg.speedCol) && isscalar(cfg.speedCol), 'cfg.speedCol required');
assert(isfield(cfg,'in23Col')   && isnumeric(cfg.in23Col)  && isscalar(cfg.in23Col),  'cfg.in23Col required');
assert(isfield(cfg,'gasName')   && ~isempty(cfg.gasName),   'cfg.gasName required');

if ~isfield(cfg,'extInputMat') || isempty(cfg.extInputMat),   cfg.extInputMat  = cfg.inputMat;  end
if ~isfield(cfg,'extOutputMat') || isempty(cfg.extOutputMat), cfg.extOutputMat = cfg.outputMat; end
if ~isfield(cfg,'modelOut') || isempty(cfg.modelOut),         cfg.modelOut     = 'turbo_model_v35.mat'; end
if ~isfield(cfg,'reportOut') || isempty(cfg.reportOut),       cfg.reportOut    = 'turbo_external_report_v35.mat'; end
if ~isfield(cfg,'K') || isempty(cfg.K),                       cfg.K            = 5; end
if ~isfield(cfg,'testFrac') || isempty(cfg.testFrac),         cfg.testFrac     = 0.15; end
if ~isfield(cfg,'etaTuneOnExternal') || isempty(cfg.etaTuneOnExternal), cfg.etaTuneOnExternal = true; end
if ~isfield(cfg,'rngSeed') || isempty(cfg.rngSeed),           cfg.rngSeed      = 42; end

fprintf('run_v35_train_and_external\n');
fprintf('[RUN] internal X mat=%s | Y mat=%s\n', char(cfg.inputMat), char(cfg.outputMat));

Xs = load(cfg.inputMat);
Ys = load(cfg.outputMat);
X = double(load_matrix_any(Xs, 'input'));
Y = double(load_matrix_any(Ys, 'output'));

fprintf('Loaded N=%d | X=%dx%d | Y=%dx%d\n', size(X,1), size(X,1), size(X,2), size(Y,1), size(Y,2));

repInt = sanity_check_ent_dsfactor_v35(X, Y, cfg);

% external data (can be same)
Xse = load(cfg.extInputMat);
Yse = load(cfg.extOutputMat);
Xext = double(load_matrix_any(Xse, 'input'));
Yext = double(load_matrix_any(Yse, 'output'));

cfgTrain = struct();
cfgTrain.X = X;
cfgTrain.Y = Y;
cfgTrain.speedCol = cfg.speedCol;
cfgTrain.in23Col  = cfg.in23Col;
cfgTrain.gasName  = cfg.gasName;
cfgTrain.modelOut = cfg.modelOut;
cfgTrain.K = cfg.K;
cfgTrain.testFrac = cfg.testFrac;
cfgTrain.rngSeed  = cfg.rngSeed;

cfgTrain.etaTuneOnExternal = cfg.etaTuneOnExternal;
cfgTrain.extX = Xext;
cfgTrain.extY = Yext;

[best, results] = train_turbocompressor_model_v35(cfgTrain); %#ok<ASGLU>

% evaluate on external
fprintf('[RUN] external X mat=%s | Y mat=%s\n', char(cfg.extInputMat), char(cfg.extOutputMat));
YhatExt = predict_turbocompressor_model_v35(cfg.modelOut, Xext);

names = ["ETA","TPR","ENT","ALPHA2","VTH2"];
reportExt = v35_metrics_table(Yext(:,1:5), YhatExt(:,1:5), names);

fprintf('==== EXTERNAL METRICS ====\n');
disp(reportExt);

save(char(cfg.reportOut), 'reportExt', 'cfg', '-v7.3');
fprintf('Saved -> %s\n', char(cfg.reportOut));

out = struct();
out.best = best;
out.results = results;
out.repInt = repInt;
out.reportExt = reportExt;
end

% ---------- helpers ----------
function T = v35_metrics_table(Y, Yhat, names)
rmse = zeros(1, size(Y,2));
mae  = zeros(1, size(Y,2));
r2   = zeros(1, size(Y,2));
for i=1:size(Y,2)
    e = Yhat(:,i) - Y(:,i);
    rmse(i) = sqrt(mean(e.^2, 'omitnan'));
    mae(i)  = mean(abs(e), 'omitnan');
    y = Y(:,i);
    sse = nansum((y - Yhat(:,i)).^2);
    sst = nansum((y - nanmean(y)).^2);
    if sst <= 0
        r2(i) = NaN;
    else
        r2(i) = 1 - sse/sst;
    end
end
T = table(names(:), rmse(:), mae(:), r2(:), 'VariableNames', {'Output','RMSE','MAE','R2'});
end
