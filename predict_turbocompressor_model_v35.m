function Yhat = predict_turbocompressor_model_v35(modelPath, X, varargin)
%PREDICT_TURBOCOMPRESSOR_MODEL_V35 Predict Nx5 outputs using turbo_v35 model.
% Usage:
%   Yhat = predict_turbocompressor_model_v35("turbo_model_v35.mat", X);

S = load(modelPath);
best = S.best;

X = double(X);
N = size(X,1);

C = best.consts;
speedCol = best.cfg.speedCol;
in23Col  = best.cfg.in23Col;

% --- TPR
tprHat = v35_predict_candidate(best.tpr, X);

% --- ETA base + bias
etaBase = v35_predict_eta_model(best.eta.base, X, tprHat);
etaData = v35_apply_poly(best.eta.bias, etaBase);

% --- DS correction (ENT pipeline)
dsPred = v35_dsfactor(etaData, tprHat, X(:,in23Col), C);
deltaDs = predict(best.ent.dsCorr.model, v35_ds_corr_features(X, etaData, tprHat, speedCol));
dsCorr = dsPred + deltaDs;

% --- ENT from corrected ds + ENT bias
entHat = best.ent.aff.a * dsCorr + best.ent.aff.b;
entHat = v35_apply_poly(best.ent.bias, entHat);

% --- ETA phys (invert dsfactor)
etaPhys = v35_eta_from_dsfactor(dsCorr, tprHat, X(:,in23Col), C);
etaPhys = v35_clip(etaPhys, 1e-6, 1-1e-6);

% --- Gate fine-tune around w0 (no collapse)
w = v35_gate_predict(best.eta.gate, X, etaData, etaPhys, tprHat, dsCorr, speedCol, in23Col);
etaHat = (1 - w).*etaData + w.*etaPhys;
etaHat = v35_clip(etaHat, 1e-6, 1-1e-6);

% --- other outputs
alpha2Hat = v35_predict_candidate(best.alpha2, X);
vth2Hat   = v35_predict_candidate(best.vth2, X);

Yhat = zeros(N,5);
Yhat(:,1) = etaHat;
Yhat(:,2) = tprHat;
Yhat(:,3) = entHat;
Yhat(:,4) = alpha2Hat;
Yhat(:,5) = vth2Hat;

end

% ---- shared helpers (same as train) ----
function yhat = v35_predict_candidate(m, X)
yT = predict(m.model, X);
yhat = m.invFcn(yT);
end

function etaHat = v35_predict_eta_model(etaM, X, tprHat)
Phi = v35_eta_features(X, tprHat, etaM.spec);
z = predict(etaM.model, Phi);
etaHat = 1./(1+exp(-z));
etaHat = v35_clip(etaHat, 1e-6, 1-1e-6);
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

function F = v35_ds_corr_features(X, etaHat, tprHat, speedCol)
spd = X(:,speedCol);
logSpd = log(max(spd, 1e-9));
logTpr = log(max(tprHat, 1e-9));
F = [X, etaHat, tprHat, logTpr, spd, logSpd];
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

function y = v35_apply_poly(bias, yHat)
if bias.deg==0
    y = yHat;
else
    y = polyval(bias.p, yHat);
end
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
