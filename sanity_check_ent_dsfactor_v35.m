function rep = sanity_check_ent_dsfactor_v35(X, Y, cfg)
%SANITY_CHECK_ENT_DSFACTOR_V35 Checks ENT vs dsfactor(TRUE) affinity.
%
% Expected:
%   ENT ~= a * dsfactor(ETA,TPR,input23) + b  with corr ~ 1 for your dataset.

C = v35_gas_consts(char(cfg.gasName));

eta = Y(:,1);
tpr = Y(:,2);
ent = Y(:,3);
in23 = X(:, cfg.in23Col);

ds = v35_dsfactor(eta, tpr, in23, C);
p = polyfit(ds, ent, 1);
a = p(1); b = p(2);

entFit = a*ds + b;
rmse = sqrt(mean((entFit - ent).^2,'omitnan'));

corrv = corr(ds, ent, 'Rows','complete');
sse = nansum((ent - entFit).^2);
sst = nansum((ent - nanmean(ent)).^2);
if sst<=0, r2 = NaN; else, r2 = 1 - sse/sst; end

fprintf('[SANITY] ENT vs dsfactor(TRUE)\n');
fprintf('  gas=%s | input23Col=%d\n', char(cfg.gasName), cfg.in23Col);
fprintf('  affine: ENT ~= a*ds + b | a=%.6g b=%.6g\n', a, b);
fprintf('  corr(ds,ENT)=%.4f | RMSE=%.6g | R2=%.6g\n', corrv, rmse, r2);

rep = struct('a',a,'b',b,'rmse',rmse,'r2',r2,'corr',corrv);
end

% ---- shared helpers ----
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
    error('Unknown gasName: %s', gasName);
end
end

function ds = v35_dsfactor(eta, tpr, in23, C)
eta = min(max(eta, 1e-6), 1-1e-6);
tpr = max(tpr, 1e-9);
den = 0.5*(0.5*in23).^2;
k = C.R / C.cp;
term = 1 + (tpr.^k - 1)./eta;
ds = C.Tt0 * (C.cp*log(term) - C.R*log(tpr)) ./ den;
end
