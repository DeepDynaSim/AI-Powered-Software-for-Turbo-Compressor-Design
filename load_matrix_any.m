function X = load_matrix_any(S, baseName)
%LOAD_MATRIX_ANY Load a numeric matrix from a MAT-loaded struct.
% Prefer S.(baseName) if it exists; otherwise pick the first numeric 2-D matrix field.

if isfield(S, baseName)
    X = S.(baseName);
    return;
end

% Common alternatives
alts = {'X','x','input','inputs','Xin','Y','y','output','outputs','Yout'};
for i = 1:numel(alts)
    nm = alts{i};
    if isfield(S, nm)
        v = S.(nm);
        if isnumeric(v) && ismatrix(v)
            X = v;
            return;
        end
    end
end

fns = fieldnames(S);
for i = 1:numel(fns)
    v = S.(fns{i});
    if isnumeric(v) && ismatrix(v)
        X = v;
        return;
    end
end

error('No numeric matrix found in MAT struct.');
end
