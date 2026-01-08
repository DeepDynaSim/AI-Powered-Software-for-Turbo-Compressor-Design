function [best, results] = train_turbocompressor_model_v35(cfg)
%TRAIN_TURBOCOMPRESSOR_MODEL_V35
% v35 focuses on ETA improvement:
% - base ETA model (CV selection)
% - ds-space correction for ENT pipeline
% - ETA-CYCLE: blend eta_data with eta_phys (from corrected dsfactor) using global w0
% - ETA-GATE: adaptive fine-tune around w0 using fixed intercept (z0) + no-intercept ridge on delta-z

% -------- input handling
if isfield(cfg,'X') && isfield(cfg,'Y')
    X = double(cfg.X);
    Y = double(cfg.Y);
else
    assert(isfield(cfg,'inputMat') && isfield(cfg,'outputMat'), 'Provide cfg.X/cfg.Y or cfg.inputMat/cfg.outputMat');
    Xs = load(cfg.inputMat); Ys = load(cfg.outputMat);
    X = double(load_matrix_any(Xs,'input'));
    Y = double(load_matrix_any(Ys,'output'));
end

assert(size(X,1)==size(Y,1), 'X/Y row mismatch');
assert(size(Y,2)>=5, 'Y must be N x 5');
N = size(X,1);

speedCol = cfg.speedCol;
in23Col  = cfg.in23Col;
gasName  = char(cfg.gasName);

if ~isfield(cfg,'K') || isempty(cfg.K), cfg.K = 5; end
if ~isfield(cfg,'testFrac') || isempty(cfg.testFrac), cfg.testFrac = 0.15; end
if ~isfield(cfg,'rngSeed') || isempty(cfg.rngSeed), cfg.rngSeed = 42; end
if ~isfield(cfg,'modelOut') || isempty(cfg.modelOut), cfg.modelOut = 'turbo_model_v35.mat'; end

if ~isfield(cfg,'etaTuneOnExternal') || isempty(cfg.etaTuneOnExternal), cfg.etaTuneOnExternal = true; end
if ~isfield(cfg,'extX'), cfg.extX = []; end
if ~isfield(cfg,'extY'), cfg.extY = []; end

rng(cfg.rngSeed);

names = ["ETA","TPR","ENT","ALPHA2","VTH2"];

% -------- constants
C = v35_gas_consts(gasName);

% -------- ENT affine sanity (ENT ~ a*ds + b)
tprTrue = Y(:,2);
etaTrue = Y(:,1);
in23    = X(:,in23Col);
dsTrue  = v35_dsfactor(etaTrue, tprTrue, in23, C);

p = polyfit(dsTrue, Y(:,3), 1);
aAff = p(1);
bAff = p(2);

% -------- split TrainVal / Test
cvHold = cvpartition(N, 'Holdout', cfg.testFrac);
idxTest = test(cvHold);
idxTV   = training(cvHold);

Xtv = X(idxTV,:);
Ytv = Y(idxTV,:);
Xte = X(idxTest,:);
Yte = Y(idxTest,:);

K = cfg.K;
cvK = cvpartition(size(Xtv,1), 'KFold', K);

fprintf('[SPLIT] TrainVal=%d | Test=%d | K=%d\n', size(Xtv,1), size(Xte,1), K);
fprintf('[ENT-PHYS] TRUE dsfactor affine: a=%.6g b=%.6g | gas=%s\n', aAff, bAff, gasName);

% -------- Train TPR (K-fold CV selection)
fprintf('[STAGE] Train TPR (K-fold CV selection)\n');
tprChoice = v35_cv_select_basic(Xtv, Ytv(:,2), cvK, 'TPR', {'gpr/identity','ridge/identity','bag/identity','lsboost/identity','gpr/log','ridge/log','bag/log','lsboost/log'});
fprintf('[TPR-CV] picked %s | cvRMSE=%.6g\n', tprChoice.name, tprChoice.cvRMSE);

% Train final TPR on TrainVal
tprModelTV = v35_fit_candidate(Xtv, Ytv(:,2), tprChoice.name);

% OOF predictions for TPR on TrainVal
tprOOF = v35_oof_predict(Xtv, Ytv(:,2), cvK, tprChoice.name);

% -------- dsTrue in ds-space (from ENT via affine invert)
dsTrueTV = (Ytv(:,3) - bAff) / aAff;

% -------- ETA selection (spec + model) using CV and adaptive lambda
fprintf('[STAGE] Train ETA (K-fold CV selection + adaptive lambda + OOF bias)\n');

% fixed subset (from your earlier forward selection)
xSubIdx = [2 3 8 10 12 14 19 23];
xSubIdx = xSubIdx(xSubIdx>=1 & xSubIdx<=size(X,2));

etaSpecs = v35_eta_specs(xSubIdx);
lambda = v35_adaptive_lambda(dsTrueTV, Ytv(:,1)); % ~0.128 typical
etaChoice = v35_cv_select_eta(Xtv, Ytv(:,1), tprOOF, dsTrueTV, cvK, etaSpecs, lambda);

fprintf('[ETA-ADAPT-CV] base=%s | lambda=%.3g | picked=%s | model=%s\n', ...
    etaChoice.baseSpec, lambda, etaChoice.spec.name, etaChoice.modelName);

% Train final ETA base on TrainVal
etaBaseTV = v35_fit_eta_model(Xtv, Ytv(:,1), tprOOF, etaChoice.spec, etaChoice.modelName);

% OOF ETA base predictions
etaBaseOOF = v35_oof_predict_eta(Xtv, Ytv(:,1), tprOOF, cvK, etaChoice.spec, etaChoice.modelName);

% Bias correction (poly degree 0..2)
[etaBias, etaOOF] = v35_fit_poly_bias(Ytv(:,1), etaBaseOOF, 'ETA');

% -------- ENT DS_CORR model (train on TrainVal using OOF eta/tpr)
fprintf('[STAGE] Train ENT (ds-space correction)\n');

dsPredOOF = v35_dsfactor(etaOOF, tprOOF, Xtv(:,in23Col), C);
deltaDsTV = dsTrueTV - dsPredOOF;

% DS_CORR selection by CV on deltaDs
dsCorrChoice = v35_cv_select_basic(v35_ds_corr_features(Xtv, etaOOF, tprOOF, speedCol), deltaDsTV, cvK, 'DS_CORR', ...
    {'gpr/identity','ridge/identity','bag/identity','lsboost/identity'});

fprintf('[DS_CORR-CV] picked %s | cvRMSE=%.6g\n', dsCorrChoice.name, dsCorrChoice.cvRMSE);

% Train DS_CORR model on TrainVal
dsCorrTV = v35_fit_candidate(v35_ds_corr_features(Xtv, etaOOF, tprOOF, speedCol), deltaDsTV, dsCorrChoice.name);

% Predict ENT on TrainVal (in ds-space)
deltaDsHatOOF = predict(dsCorrTV.model, v35_ds_corr_features(Xtv, etaOOF, tprOOF, speedCol));
dsCorrOOF = dsPredOOF + deltaDsHatOOF;
entPredOOF = aAff*dsCorrOOF + bAff;

% ENT bias correction
[entBias, entOOF] = v35_fit_poly_bias(Ytv(:,3), entPredOOF, 'ENT'); %#ok<ASGLU>

% -------- Other outputs (ALPHA2, VTH2) CV pick (simple)
fprintf('[STAGE] Train ALPHA2 & VTH2 (K-fold CV selection)\n');

a2Choice = v35_cv_select_basic(Xtv, Ytv(:,4), cvK, 'ALPHA2', {'gpr/identity','ridge/identity','bag/identity','lsboost/identity'});
fprintf('[ALPHA2-CV] picked %s | cvRMSE=%.6g\n', a2Choice.name, a2Choice.cvRMSE);
a2ModelTV = v35_fit_candidate(Xtv, Ytv(:,4), a2Choice.name);
a2OOF = v35_oof_predict(Xtv, Ytv(:,4), cvK, a2Choice.name);

vthChoice = v35_cv_select_basic(Xtv, Ytv(:,5), cvK, 'VTH2', {'gpr/identity','ridge/identity','bag/identity','lsboost/identity'});
fprintf('[VTH2-CV] picked %s | cvRMSE=%.6g\n', vthChoice.name, vthChoice.cvRMSE);
vthModelTV = v35_fit_candidate(Xtv, Ytv(:,5), vthChoice.name);
vthOOF = v35_oof_predict(Xtv, Ytv(:,5), cvK, vthChoice.name);

% -------- ETA-CYCLE (global w0), then ETA-GATE (delta-z fine tune)
% Compute eta_phys from dsCorrOOF analytically (invert dsfactor)
etaPhysOOF = v35_eta_from_dsfactor(dsCorrOOF, tprOOF, Xtv(:,in23Col), C);
etaPhysOOF = v35_clip(etaPhysOOF, 1e-4, 1-1e-4);

% pick w0 (prefer external if allowed and provided)
gateCfg = v35_default_gate_cfg();
gateCfg.wMax = 0.70;
gateCfg.wMin = 0.00;

useExternalTune = cfg.etaTuneOnExternal && ~isempty(cfg.extX) && ~isempty(cfg.extY) ...
    && size(cfg.extX,1)==size(cfg.extY,1) && size(cfg.extY,2)>=2;

if useExternalTune
    % build external predictions using TrainVal-trained components (not full refit yet)
    Xext = double(cfg.extX); Yext = double(cfg.extY);
    tprExt = v35_predict_candidate(tprModelTV, Xext);
    etaExtBase = v35_predict_eta_model(etaBaseTV, Xext, tprExt);
    etaExt = v35_apply_poly(etaBias, etaExtBase);

    dsPredExt = v35_dsfactor(etaExt, tprExt, Xext(:,in23Col), C);
    deltaDsExt = predict(dsCorrTV.model, v35_ds_corr_features(Xext, etaExt, tprExt, speedCol));
    dsCorrExt = dsPredExt + deltaDsExt;

    etaPhysExt = v35_eta_from_dsfactor(dsCorrExt, tprExt, Xext(:,in23Col), C);
    etaPhysExt = v35_clip(etaPhysExt, 1e-4, 1-1e-4);

    [w0, rm0, rm1] = v35_tune_w0(Yext(:,1), etaExt, etaPhysExt, gateCfg.wMin, gateCfg.wMax);
    fprintf('[ETA-CYCLE] tuned w0=%.3f | rmse %.6g -> %.6g (external)\n', w0, rm0, rm1);
else
    [w0, rm0, rm1] = v35_tune_w0(Ytv(:,1), etaOOF, etaPhysOOF, gateCfg.wMin, gateCfg.wMax);
    fprintf('[ETA-CYCLE] tuned w0=%.3f | rmse %.6g -> %.6g (trainval OOF)\n', w0, rm0, rm1);
end

% Train gate on TrainVal (OOF-based), as *fine tune* around w0
gateCfg.w0 = w0;
gate = v35_train_eta_gate( ...
    Xtv, Ytv(:,1), etaOOF, etaPhysOOF, tprOOF, dsCorrOOF, speedCol, in23Col, gateCfg);

% Compute gated ETA OOF (for reporting)
wOOF = v35_gate_predict(gate, Xtv, etaOOF, etaPhysOOF, tprOOF, dsCorrOOF, speedCol, in23Col);
etaGatedOOF = (1 - wOOF).*etaOOF + wOOF.*etaPhysOOF;
etaGatedOOF = v35_clip(etaGatedOOF, 1e-4, 1-1e-4);

% Replace ETA OOF for metrics
etaOOF_final = etaGatedOOF;

% -------- Metrics: TrainVal OOF
YhatOOF = zeros(size(Ytv,1), 5);
YhatOOF(:,1) = etaOOF_final;
YhatOOF(:,2) = tprOOF;
YhatOOF(:,3) = v35_apply_poly(entBias, entPredOOF);
YhatOOF(:,4) = a2OOF;
YhatOOF(:,5) = vthOOF;

metricsOOF = v35_metrics_table(Ytv(:,1:5), YhatOOF(:,1:5), names);

fprintf('==== INTERNAL (TRAINVAL OOF) METRICS ====\n');
disp(metricsOOF);

% -------- Evaluate on Holdout Test (train models on TrainVal and predict Test)
tprTe = v35_predict_candidate(tprModelTV, Xte);
etaTeBase = v35_predict_eta_model(etaBaseTV, Xte, tprTe);
etaTe = v35_apply_poly(etaBias, etaTeBase);

dsPredTe = v35_dsfactor(etaTe, tprTe, Xte(:,in23Col), C);
deltaDsTe = predict(dsCorrTV.model, v35_ds_corr_features(Xte, etaTe, tprTe, speedCol));
dsCorrTe = dsPredTe + deltaDsTe;

etaPhysTe = v35_eta_from_dsfactor(dsCorrTe, tprTe, Xte(:,in23Col), C);
etaPhysTe = v35_clip(etaPhysTe, 1e-4, 1-1e-4);

wTe = v35_gate_predict(gate, Xte, etaTe, etaPhysTe, tprTe, dsCorrTe, speedCol, in23Col);
etaTeFinal = (1 - wTe).*etaTe + wTe.*etaPhysTe;
etaTeFinal = v35_clip(etaTeFinal, 1e-4, 1-1e-4);

entTePred = aAff*(dsCorrTe) + bAff;
entTePred = v35_apply_poly(entBias, entTePred);

a2Te = v35_predict_candidate(a2ModelTV, Xte);
vthTe = v35_predict_candidate(vthModelTV, Xte);

YhatTe = [etaTeFinal, tprTe, entTePred, a2Te, vthTe];
metricsTest = v35_metrics_table(Yte(:,1:5), YhatTe(:,1:5), names);

fprintf('==== INTERNAL (HOLDOUT TEST) METRICS ====\n');
disp(metricsTest);

% -------- Refit FINAL models on full data for deployment (same choices)
tprModelAll = v35_fit_candidate(X, Y(:,2), tprChoice.name);

tprAllHat = v35_predict_candidate(tprModelAll, X);
etaBaseAll = v35_fit_eta_model(X, Y(:,1), tprAllHat, etaChoice.spec, etaChoice.modelName);

etaAllBaseHat = v35_predict_eta_model(etaBaseAll, X, tprAllHat);
etaAllHat = v35_apply_poly(etaBias, etaAllBaseHat);

dsTrueAll = (Y(:,3) - bAff) / aAff;
dsPredAll = v35_dsfactor(etaAllHat, tprAllHat, X(:,in23Col), C);
deltaDsAll = dsTrueAll - dsPredAll;

dsCorrAll = v35_fit_candidate(v35_ds_corr_features(X, etaAllHat, tprAllHat, speedCol), deltaDsAll, dsCorrChoice.name);

a2ModelAll  = v35_fit_candidate(X, Y(:,4), a2Choice.name);
vthModelAll = v35_fit_candidate(X, Y(:,5), vthChoice.name);

best = struct();
best.type = 'turbo_v35';
best.cfg = cfg;

best.consts = C;

best.ent = struct();
best.ent.aff = struct('a', aAff, 'b', bAff);
best.ent.dsCorr = dsCorrAll;
best.ent.bias = entBias;

best.tpr = tprModelAll;

best.eta = struct();
best.eta.choice = etaChoice;
best.eta.base = etaBaseAll;
best.eta.bias = etaBias;
best.eta.w0 = w0;
best.eta.gate = gate;

best.alpha2 = a2ModelAll;
best.vth2   = vthModelAll;

results = struct();
results.oof = metricsOOF;
results.test = metricsTest;
results.w0 = w0;

save(char(cfg.modelOut), 'best', 'results', '-v7.3');
fprintf('Saved -> %s | type=%s\n', char(cfg.modelOut), best.type);

end

% =========================== helpers ===========================

function C = v35_gas_consts(gasName)
gasName = lower(strtrim(gasName));
C = struct();
C.Tt0 = 288.15;
if strcmp(gasName,'air')
    C.R  = 287;
    C.cp = 1005;
elseif strcmp(gasName,'hydrogen')
    C.R  = 4124;
    C.cp = 14310;
else
    error('Unknown gasName: %s (use air/hydrogen)', gasName);
end
end

function ds = v35_dsfactor(eta, tpr, in23, C)
eta = v35_clip(eta, 1e-6, 1-1e-6);
tpr = max(tpr, 1e-9);

den = 0.5*(0.5*in23).^2;
k = C.R / C.cp;

term = 1 + (tpr.^k - 1)./eta;
ds = C.Tt0 * (C.cp*log(term) - C.R*log(tpr)) ./ den;
end

function eta = v35_eta_from_dsfactor(ds, tpr, in23, C)
tpr = max(tpr, 1e-9);
den = 0.5*(0.5*in23).^2;
k = C.R / C.cp;

Cterm = tpr.^k - 1;
A = (ds .* den ./ C.Tt0) + C.R*log(tpr);
D = exp(A ./ C.cp);
eta = Cterm ./ (D - 1);

eta = v35_clip(eta, 1e-6, 1-1e-6);
end

function x = v35_clip(x, lo, hi)
x = min(max(x, lo), hi);
end

function specs = v35_eta_specs(xSubIdx)
specs = {};
specs{end+1} = struct('name','Xsub + TPR + SPD + LOGSPD',       'id',1, 'xSub',xSubIdx);
specs{end+1} = struct('name','Xsub + LOGTPR + SPD + LOGSPD',    'id',2, 'xSub',xSubIdx);
specs{end+1} = struct('name','Xsub + TPR + SPD + SPD2',         'id',3, 'xSub',xSubIdx);
specs{end+1} = struct('name','Xsub + LOGTPR + SPD + SPD2',      'id',4, 'xSub',xSubIdx);
specs{end+1} = struct('name','Xsub + TPR + SPD + SPD2 + TPRxSPD','id',5,'xSub',xSubIdx);
specs{end+1} = struct('name','Xsub + LOGTPR + LOGSPD + SPD2',   'id',6, 'xSub',xSubIdx);
end

function lambda = v35_adaptive_lambda(dsTrue, etaTrue)
sdDs  = std(dsTrue,  'omitnan');
sdEta = std(etaTrue, 'omitnan');
if sdEta <= 0 || isnan(sdEta) || isnan(sdDs)
    lambda = 0.1;
else
    lambdaRaw = sdDs / sdEta;
    lambda = lambdaRaw / 8;
end
lambda = min(max(lambda, 0.01), 0.5);
end

function choice = v35_cv_select_basic(X, y, cvK, tag, candList)
bestRm = inf;
bestName = '';
for i=1:numel(candList)
    nm = candList{i};
    yhat = v35_oof_predict(X, y, cvK, nm);
    rm = sqrt(mean((yhat - y).^2, 'omitnan'));
    fprintf('[ZOO %s] cand=%s | cvRMSE=%.6g\n', tag, nm, rm);
    if rm < bestRm
        bestRm = rm;
        bestName = nm;
    end
end
choice = struct('name', bestName, 'cvRMSE', bestRm);
end

function choice = v35_cv_select_eta(Xtv, yEta, tprOOF, dsTrueTV, cvK, etaSpecs, lambda) %#ok<INUSD>
modelNames = {'gpr','ridge','bag','lsboost'};
bestRm = inf;
best = struct();

baseSpec = etaSpecs{2}.name; % for logging

for s=1:numel(etaSpecs)
    spec = etaSpecs{s};
    for m=1:numel(modelNames)
        mdlName = modelNames{m};
        etaHat = v35_oof_predict_eta(Xtv, yEta, tprOOF, cvK, spec, mdlName);
        rmEta = sqrt(mean((etaHat - yEta).^2, 'omitnan'));
        fprintf('[ETA-CV] spec=%s | model=%s | rmse_eta=%.6g\n', spec.name, mdlName, rmEta);
        if rmEta < bestRm
            bestRm = rmEta;
            best.spec = spec;
            best.modelName = mdlName;
            best.rmse_eta = rmEta;
        end
    end
end

choice = best;
choice.baseSpec = baseSpec;
end

function m = v35_fit_candidate(X, y, candName)
parts = strsplit(candName, '/');
algo = parts{1};
xfm  = parts{2};

m = struct();
m.name = candName;
m.algo = algo;
m.xfm  = xfm;

[yT, invFcn] = v35_apply_y_transform(y, xfm);

switch lower(algo)
    case 'gpr'
        mdl = fitrgp(X, yT, ...
            'KernelFunction','ardsquaredexponential', ...
            'BasisFunction','constant', ...
            'Standardize',true);
    case 'ridge'
        mdl = fitrlinear(X, yT, ...
            'Learner','leastsquares', ...
            'Regularization','ridge', ...
            'Lambda',1e-3, ...
            'Solver','lbfgs');
    case 'bag'
        t = templateTree('MinLeafSize', 10);
        mdl = fitrensemble(X, yT, 'Method','Bag', 'Learners',t, 'NumLearningCycles', 150);
    case 'lsboost'
        t = templateTree('MinLeafSize', 8);
        mdl = fitrensemble(X, yT, 'Method','LSBoost', 'Learners',t, 'NumLearningCycles', 300, 'LearnRate', 0.05);
    otherwise
        error('Unknown algo: %s', algo);
end

m.model = mdl;
m.invFcn = invFcn;
end

function yhat = v35_predict_candidate(m, X)
yT = predict(m.model, X);
yhat = m.invFcn(yT);
end

function yhat = v35_oof_predict(X, y, cvK, candName)
yhat = nan(size(y));
for k=1:cvK.NumTestSets
    tr = training(cvK, k);
    va = test(cvK, k);
    mk = v35_fit_candidate(X(tr,:), y(tr), candName);
    yhat(va) = v35_predict_candidate(mk, X(va,:));
end
end

function [yT, invFcn] = v35_apply_y_transform(y, xfm)
switch lower(xfm)
    case 'identity'
        yT = y;
        invFcn = @(z) z;
    case 'log'
        yT = log(max(y, 1e-9));
        invFcn = @(z) exp(z);
    otherwise
        error('Unknown transform: %s', xfm);
end
end

function etaM = v35_fit_eta_model(X, yEta, tprHat, spec, modelName)
Phi = v35_eta_features(X, tprHat, spec);
y = v35_clip(yEta, 1e-4, 1-1e-4);
yT = log(y./(1-y));

switch lower(modelName)
    case 'gpr'
        mdl = fitrgp(Phi, yT, 'KernelFunction','ardsquaredexponential', 'BasisFunction','constant', 'Standardize',true);
    case 'ridge'
        mdl = fitrlinear(Phi, yT, 'Learner','leastsquares','Regularization','ridge','Lambda',1e-3,'Solver','lbfgs');
    case 'bag'
        t = templateTree('MinLeafSize', 10);
        mdl = fitrensemble(Phi, yT, 'Method','Bag','Learners',t,'NumLearningCycles',200);
    case 'lsboost'
        t = templateTree('MinLeafSize', 8);
        mdl = fitrensemble(Phi, yT, 'Method','LSBoost','Learners',t,'NumLearningCycles',400,'LearnRate',0.05);
    otherwise
        error('Unknown ETA model: %s', modelName);
end

etaM = struct();
etaM.spec = spec;
etaM.modelName = modelName;
etaM.model = mdl;
end

function etaHat = v35_predict_eta_model(etaM, X, tprHat)
Phi = v35_eta_features(X, tprHat, etaM.spec);
z = predict(etaM.model, Phi);
etaHat = 1./(1+exp(-z));
etaHat = v35_clip(etaHat, 1e-6, 1-1e-6);
end

function etaHat = v35_oof_predict_eta(X, yEta, tprOOF, cvK, spec, modelName)
etaHat = nan(size(yEta));
for k=1:cvK.NumTestSets
    tr = training(cvK,k);
    va = test(cvK,k);

    etaM = v35_fit_eta_model(X(tr,:), yEta(tr), tprOOF(tr), spec, modelName);
    etaHat(va) = v35_predict_eta_model(etaM, X(va,:), tprOOF(va));
end
end

function Phi = v35_eta_features(X, tpr, spec)
xSub = X(:, spec.xSub);

spd = X(:,1);
logSpd = log(max(spd, 1e-9));

logTpr = log(max(tpr, 1e-9));
spd2 = spd.^2;
tprxspd = tpr.*spd;

switch spec.id
    case 1
        Phi = [xSub, tpr, spd, logSpd];
    case 2
        Phi = [xSub, logTpr, spd, logSpd];
    case 3
        Phi = [xSub, tpr, spd, spd2];
    case 4
        Phi = [xSub, logTpr, spd, spd2];
    case 5
        Phi = [xSub, tpr, spd, spd2, tprxspd];
    case 6
        Phi = [xSub, logTpr, logSpd, spd2];
    otherwise
        error('Unknown ETA spec id');
end
end

function [bias, yCorr] = v35_fit_poly_bias(yTrue, yHat, tag)
degList = [0 1 2];
bestRm = inf;
best = struct('deg',0,'p',1);

for d=degList
    if d==0
        p = [1 0];
        yC = yHat;
    elseif d==1
        p = polyfit(yHat, yTrue, 1);
        yC = polyval(p, yHat);
    else
        p = polyfit(yHat, yTrue, 2);
        yC = polyval(p, yHat);
    end
    rm = sqrt(mean((yC - yTrue).^2, 'omitnan'));
    if rm < bestRm
        bestRm = rm;
        best.deg = d;
        best.p = p;
    end
end

bias = best;
yCorr = v35_apply_poly(bias, yHat);

fprintf('[BIAS %s] deg=%d | rmse=%.6g\n', tag, bias.deg, bestRm);
end

function y = v35_apply_poly(bias, yHat)
d = bias.deg;
if d==0
    y = yHat;
else
    y = polyval(bias.p, yHat);
end
end

function F = v35_ds_corr_features(X, etaHat, tprHat, speedCol)
spd = X(:,speedCol);
logSpd = log(max(spd, 1e-9));
logTpr = log(max(tprHat, 1e-9));
F = [X, etaHat, tprHat, logTpr, spd, logSpd];
end

function gateCfg = v35_default_gate_cfg()
gateCfg = struct();
gateCfg.wMin = 0.0;
gateCfg.wMax = 0.7;
gateCfg.w0   = 0.4;
gateCfg.deltaZClip = 0.75;
gateCfg.ridgeLambda = 1e-2;
end

function [w0, rm0, rm1] = v35_tune_w0(yTrue, etaData, etaPhys, wMin, wMax)
grid = 0:0.02:wMax;
rm0 = sqrt(mean((etaData - yTrue).^2,'omitnan'));
bestRm = inf; bestW = 0;
for w = grid
    y = (1-w).*etaData + w.*etaPhys;
    rm = sqrt(mean((y - yTrue).^2,'omitnan'));
    if rm < bestRm
        bestRm = rm;
        bestW = w;
    end
end
w0 = min(max(bestW, wMin), wMax);
rm1 = bestRm;
end

function gate = v35_train_eta_gate(X, yTrue, etaData, etaPhys, tprHat, dsCorr, speedCol, in23Col, gateCfg)
wMin = gateCfg.wMin;
wMax = gateCfg.wMax;
w0   = min(max(gateCfg.w0, wMin+1e-6), wMax-1e-6);

den = (etaPhys - etaData);
wStar = w0*ones(size(yTrue));
ok = abs(den) > 1e-6;
wStar(ok) = (yTrue(ok) - etaData(ok)) ./ den(ok);
wStar = v35_clip(wStar, wMin, wMax);

z0 = log((w0 - wMin) / (wMax - w0));
zStar = log((wStar - wMin) ./ (wMax - wStar));
deltaZ = zStar - z0;

spd = X(:,speedCol);
in23 = X(:,in23Col);
phi = [ ...
    spd, ...
    log(max(spd,1e-9)), ...
    in23, ...
    log(max(in23,1e-9)), ...
    tprHat, ...
    log(max(tprHat,1e-9)), ...
    etaData, ...
    abs(etaPhys - etaData), ...
    dsCorr ...
];

mu = mean(phi,1,'omitnan');
sg = std(phi,0,1,'omitnan'); sg(sg<1e-12) = 1;
phiZ = (phi - mu) ./ sg;

lambda = gateCfg.ridgeLambda;
A = (phiZ' * phiZ) + lambda*eye(size(phiZ,2));
b = (phiZ' * deltaZ);
beta = A \ b;

gate = struct();
gate.wMin = wMin;
gate.wMax = wMax;
gate.w0   = w0;
gate.z0   = z0;
gate.mu   = mu;
gate.sg   = sg;
gate.beta = beta;
gate.deltaZClip = gateCfg.deltaZClip;

deltaZhat = phiZ * beta;
deltaZhat = v35_clip(deltaZhat, -gate.deltaZClip, gate.deltaZClip);
wHat = wMin + (wMax - wMin) ./ (1 + exp(-(z0 + deltaZhat)));
rm0 = sqrt(mean(((1-w0).*etaData + w0.*etaPhys - yTrue).^2,'omitnan'));
rm1 = sqrt(mean(((1-wHat).*etaData + wHat.*etaPhys - yTrue).^2,'omitnan'));
fprintf('[ETA-GATE] fine-tune around w0=%.3f | rmse %.6g -> %.6g (trainval)\n', w0, rm0, rm1);
end

function w = v35_gate_predict(gate, X, etaData, etaPhys, tprHat, dsCorr, speedCol, in23Col)
spd = X(:,speedCol);
in23 = X(:,in23Col);
phi = [ ...
    spd, ...
    log(max(spd,1e-9)), ...
    in23, ...
    log(max(in23,1e-9)), ...
    tprHat, ...
    log(max(tprHat,1e-9)), ...
    etaData, ...
    abs(etaPhys - etaData), ...
    dsCorr ...
];
phiZ = (phi - gate.mu) ./ gate.sg;
deltaZ = phiZ * gate.beta;
deltaZ = v35_clip(deltaZ, -gate.deltaZClip, gate.deltaZClip);

z = gate.z0 + deltaZ;
w = gate.wMin + (gate.wMax - gate.wMin) ./ (1 + exp(-z));
w = v35_clip(w, gate.wMin, gate.wMax);
end

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
