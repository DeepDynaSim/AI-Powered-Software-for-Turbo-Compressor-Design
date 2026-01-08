% ========================================================================
% File: src/turbo_results_dashboard_ui_v35.m
% Turbo compressor results dashboard (v35+ FIXED)
%
% - Load model + external input/output MAT
% - Predict via predictFcn(modelPath, X) (fallback: (modelPath,X,false))
% - Compute RMSE/MAE/R2 + shift stats
% - Plots: parity, residual vs speed, histogram, series
% - ETA Focus: speed-bin metrics + plots
% - Export: metrics.csv, shift.csv, eta_bins.csv, rows_preview.csv,
%          per-output PNG + multipage PDF report
% ========================================================================
function turbo_results_dashboard_ui_v35(varargin)

p = inputParser;
p.addParameter("modelPath","", @(s)ischar(s)||isstring(s));
p.addParameter("extInput","",  @(s)ischar(s)||isstring(s));
p.addParameter("extOutput","", @(s)ischar(s)||isstring(s));
p.addParameter("predictFcn","predict_turbocompressor_model_v35", @(s)ischar(s)||isstring(s));
p.addParameter("speedIdx",23, @(x)isnumeric(x)&&isscalar(x)&&x>=1);
p.addParameter("outputNames",["ETA","TPR","ENT","ALPHA2","VTH2"], @(x)isstring(x)||iscellstr(x));
p.addParameter("etaBins",12, @(x)isnumeric(x)&&isscalar(x)&&x>=3);
p.parse(varargin{:});
cfg = p.Results;

state = struct();
state.modelPath   = string(cfg.modelPath);
state.extInput    = string(cfg.extInput);
state.extOutput   = string(cfg.extOutput);
state.predictFcn  = string(cfg.predictFcn);
state.speedIdx    = double(cfg.speedIdx);
state.outputNames = string(cfg.outputNames(:))';
state.etaBins     = double(cfg.etaBins);

state.X = [];
state.Y = [];
state.Yhat = [];
state.metrics = table();
state.shift = table();
state.detailTable = table();
state.etaBinTable = table();

% ---------------- UI root
fig = uifigure("Name","Turbo Results Dashboard (v35+)","Position",[100 100 1480 820]);

root = uigridlayout(fig,[1 2]);
root.ColumnWidth = {420,'1x'};
root.RowHeight = {'1x'};

% ---------------- Left panel
left = uipanel(root,"Title","Controls");
left.Layout.Row = 1; left.Layout.Column = 1;

lg = uigridlayout(left,[20 2]);
lg.Padding = [10 10 10 10];
lg.RowSpacing = 8;
lg.ColumnSpacing = 10;
lg.ColumnWidth = {'1x',120};
lg.RowHeight = { ...
    22, 28, ... % model
    22, 28, ... % xin
    22, 28, ... % yout
    22, 28, ... % predict fcn
    22, 28, ... % speed idx
    22, 28, ... % eta bins
    34, ...     % run
    22, 28, ... % output dropdown
    92, ...     % mini cards
    34, 34, ... % export buttons
    22, ...     % tips
    '1x', ...   % summary
    22, ...     % status
    34};        % close

% Row 1-2: Model
lbl = uilabel(lg,"Text","Model (.mat)","FontWeight","bold");
lbl.Layout.Row = 1; lbl.Layout.Column = [1 2];
edtModel = uieditfield(lg,"text","Value",char(state.modelPath));
edtModel.Layout.Row = 2; edtModel.Layout.Column = 1;
btn = uibutton(lg,"Text","Browse","ButtonPushedFcn",@onBrowseModel);
btn.Layout.Row = 2; btn.Layout.Column = 2;

% Row 3-4: External input
lbl = uilabel(lg,"Text","External Input (.mat)","FontWeight","bold");
lbl.Layout.Row = 3; lbl.Layout.Column = [1 2];
edtXin = uieditfield(lg,"text","Value",char(state.extInput));
edtXin.Layout.Row = 4; edtXin.Layout.Column = 1;
btn = uibutton(lg,"Text","Browse","ButtonPushedFcn",@onBrowseXin);
btn.Layout.Row = 4; btn.Layout.Column = 2;

% Row 5-6: External output
lbl = uilabel(lg,"Text","External Output (.mat)","FontWeight","bold");
lbl.Layout.Row = 5; lbl.Layout.Column = [1 2];
edtYout = uieditfield(lg,"text","Value",char(state.extOutput));
edtYout.Layout.Row = 6; edtYout.Layout.Column = 1;
btn = uibutton(lg,"Text","Browse","ButtonPushedFcn",@onBrowseYout);
btn.Layout.Row = 6; btn.Layout.Column = 2;

% Row 7-8: Predict function
lbl = uilabel(lg,"Text","Predict Function","FontWeight","bold");
lbl.Layout.Row = 7; lbl.Layout.Column = [1 2];
edtPredFcn = uieditfield(lg,"text","Value",char(state.predictFcn));
edtPredFcn.Layout.Row = 8; edtPredFcn.Layout.Column = 1;
btn = uibutton(lg,"Text","Check","ButtonPushedFcn",@onCheckFcn);
btn.Layout.Row = 8; btn.Layout.Column = 2;

% Row 9-10: Speed index
lbl = uilabel(lg,"Text","Speed Index (X(:,speedIdx))","FontWeight","bold");
lbl.Layout.Row = 9; lbl.Layout.Column = [1 2];
edtSpeedIdx = uieditfield(lg,"numeric","Value",state.speedIdx,"Limits",[1 Inf]);
edtSpeedIdx.Layout.Row = 10; edtSpeedIdx.Layout.Column = 1;
btn = uibutton(lg,"Text","Set","ButtonPushedFcn",@onSetSpeedIdx);
btn.Layout.Row = 10; btn.Layout.Column = 2;

% Row 11-12: ETA bins
lbl = uilabel(lg,"Text","ETA Speed Bins","FontWeight","bold");
lbl.Layout.Row = 11; lbl.Layout.Column = [1 2];
edtEtaBins = uieditfield(lg,"numeric","Value",state.etaBins,"Limits",[3 100]);
edtEtaBins.Layout.Row = 12; edtEtaBins.Layout.Column = 1;
btn = uibutton(lg,"Text","Set","ButtonPushedFcn",@onSetEtaBins);
btn.Layout.Row = 12; btn.Layout.Column = 2;

% Row 13: Run
btnRun = uibutton(lg,"Text","Load & Evaluate","FontWeight","bold","ButtonPushedFcn",@onRun);
btnRun.Layout.Row = 13; btnRun.Layout.Column = [1 2];

% Row 14-15: Output dropdown
lbl = uilabel(lg,"Text","Output Select","FontWeight","bold");
lbl.Layout.Row = 14; lbl.Layout.Column = [1 2];
ddOut = uidropdown(lg,"Items",cellstr(state.outputNames), "Value", char(state.outputNames(1)), ...
    "ValueChangedFcn",@onSelectOutput);
ddOut.Layout.Row = 15; ddOut.Layout.Column = [1 2];

% Row 16: Mini metric cards
cardPanel = uipanel(lg,"Title","Selected Output Metrics");
cardPanel.Layout.Row = 16; cardPanel.Layout.Column = [1 2];
cg = uigridlayout(cardPanel,[2 3]);
cg.RowHeight = {22,22};
cg.ColumnWidth = {'1x','1x','1x'};
cg.Padding = [8 8 8 8];
cg.RowSpacing = 6;

lbl = uilabel(cg,"Text","RMSE","FontWeight","bold"); lbl.Layout.Row=1; lbl.Layout.Column=1;
lbl = uilabel(cg,"Text","MAE","FontWeight","bold");  lbl.Layout.Row=1; lbl.Layout.Column=2;
lbl = uilabel(cg,"Text","R2","FontWeight","bold");   lbl.Layout.Row=1; lbl.Layout.Column=3;

valCardRMSE = uilabel(cg,"Text","-"); valCardRMSE.Layout.Row=2; valCardRMSE.Layout.Column=1;
valCardMAE  = uilabel(cg,"Text","-"); valCardMAE.Layout.Row=2;  valCardMAE.Layout.Column=2;
valCardR2   = uilabel(cg,"Text","-"); valCardR2.Layout.Row=2;   valCardR2.Layout.Column=3;

% Row 17-18: Export buttons
btnExportCSV = uibutton(lg,"Text","Export Metrics CSV","ButtonPushedFcn",@onExportCSV);
btnExportCSV.Layout.Row = 17; btnExportCSV.Layout.Column = [1 2];

btnExportReport = uibutton(lg,"Text","Export Project Report (PDF+PNG)","FontWeight","bold","ButtonPushedFcn",@onExportReport);
btnExportReport.Layout.Row = 18; btnExportReport.Layout.Column = [1 2];

% Row 19: Tips
lblTips = uilabel(lg,"Text","Tip: External is evaluation-only for fair reporting.","FontWeight","bold");
lblTips.Layout.Row = 19; lblTips.Layout.Column = [1 2];

% Row 20: Summary
txtSummary = uitextarea(lg,"Editable","off","Value", ...
    ["Ready."; "1) Select files"; "2) Load & Evaluate"; "3) Export report"]);
txtSummary.Layout.Row = 20; txtSummary.Layout.Column = [1 2];

% Status + Close (on figure)
lblStatus = uilabel(fig,"Text","Status: idle","FontWeight","bold","Position",[110 10 850 28]);
uibutton(fig,"Text","Close","Position",[10 10 80 28],"ButtonPushedFcn",@(s,e)delete(fig));

% ---------------- Right panel
right = uipanel(root,"Title","Results");
right.Layout.Row = 1; right.Layout.Column = 2;

rg = uigridlayout(right,[1 1]);
rg.RowHeight = {'1x'};
rg.ColumnWidth = {'1x'};
rg.Padding = [10 10 10 10];

tg = uitabgroup(rg);
tg.Layout.Row = 1; tg.Layout.Column = 1;

tab1 = uitab(tg,"Title","Metrics");
tab2 = uitab(tg,"Title","Plots");
tab3 = uitab(tg,"Title","Row List");
tab4 = uitab(tg,"Title","ETA Focus");

% Metrics tab
g1 = uigridlayout(tab1,[3 1]);
g1.RowHeight = {260, 210, '1x'};
g1.ColumnWidth = {'1x'};

tblMetrics = uitable(g1,"Data",table(),"ColumnEditable",false);
tblMetrics.Layout.Row = 1; tblMetrics.Layout.Column = 1;

tblShift = uitable(g1,"Data",table(),"ColumnEditable",false);
tblShift.Layout.Row = 2; tblShift.Layout.Column = 1;

tblEtaBins = uitable(g1,"Data",table(),"ColumnEditable",false);
tblEtaBins.Layout.Row = 3; tblEtaBins.Layout.Column = 1;

% Plots tab
g2 = uigridlayout(tab2,[2 2]);
g2.RowHeight = {'1x','1x'};
g2.ColumnWidth = {'1x','1x'};

axParity = uiaxes(g2); axParity.Layout.Row=1; axParity.Layout.Column=1;
title(axParity,"Parity (True vs Pred)"); grid(axParity,"on"); xlabel(axParity,"True"); ylabel(axParity,"Pred");

axResSpeed = uiaxes(g2); axResSpeed.Layout.Row=1; axResSpeed.Layout.Column=2;
title(axResSpeed,"Residual vs Speed"); grid(axResSpeed,"on"); xlabel(axResSpeed,"Speed"); ylabel(axResSpeed,"Pred - True");

axHist = uiaxes(g2); axHist.Layout.Row=2; axHist.Layout.Column=1;
title(axHist,"Residual Histogram"); grid(axHist,"on"); xlabel(axHist,"Residual"); ylabel(axHist,"Count");

axSeries = uiaxes(g2); axSeries.Layout.Row=2; axSeries.Layout.Column=2;
title(axSeries,"True & Pred (index)"); grid(axSeries,"on"); xlabel(axSeries,"Row index"); ylabel(axSeries,"Value");

% Row list tab
g3 = uigridlayout(tab3,[2 1]);
g3.RowHeight = {48,'1x'};
g3.ColumnWidth = {'1x'};

slN = uislider(g3,"Limits",[10 5000],"Value",200,"ValueChangedFcn",@onNRowsChanged);
slN.Layout.Row=1; slN.Layout.Column=1;
slN.MajorTicks = [10 100 200 500 1000 2000 5000];

tblRows = uitable(g3,"Data",table(),"ColumnEditable",false);
tblRows.Layout.Row=2; tblRows.Layout.Column=1;

% ETA Focus tab
g4 = uigridlayout(tab4,[2 2]);
g4.RowHeight = {'1x','1x'};
g4.ColumnWidth = {'1x','1x'};

axEtaBin = uiaxes(g4); axEtaBin.Layout.Row=1; axEtaBin.Layout.Column=1;
title(axEtaBin,"ETA RMSE by Speed Bin"); grid(axEtaBin,"on"); xlabel(axEtaBin,"Bin center speed"); ylabel(axEtaBin,"RMSE");

axEtaResPred = uiaxes(g4); axEtaResPred.Layout.Row=1; axEtaResPred.Layout.Column=2;
title(axEtaResPred,"ETA Residual vs Pred"); grid(axEtaResPred,"on"); xlabel(axEtaResPred,"Pred ETA"); ylabel(axEtaResPred,"Residual");

axEtaResSpeed = uiaxes(g4); axEtaResSpeed.Layout.Row=2; axEtaResSpeed.Layout.Column=1;
title(axEtaResSpeed,"ETA Residual vs Speed"); grid(axEtaResSpeed,"on"); xlabel(axEtaResSpeed,"Speed"); ylabel(axEtaResSpeed,"Residual");

axEtaParity = uiaxes(g4); axEtaParity.Layout.Row=2; axEtaParity.Layout.Column=2;
title(axEtaParity,"ETA Parity"); grid(axEtaParity,"on"); xlabel(axEtaParity,"True"); ylabel(axEtaParity,"Pred");

% Auto-run
if strlength(state.modelPath) > 0 && strlength(state.extInput) > 0 && strlength(state.extOutput) > 0
    drawnow;
    onRun();
end

% ===================== Callbacks =====================
    function onBrowseModel(~,~)
        [f,pth] = uigetfile("*.mat","Select Model MAT");
        if isequal(f,0), return; end
        edtModel.Value = fullfile(pth,f);
    end

    function onBrowseXin(~,~)
        [f,pth] = uigetfile("*.mat","Select External Input MAT");
        if isequal(f,0), return; end
        edtXin.Value = fullfile(pth,f);
    end

    function onBrowseYout(~,~)
        [f,pth] = uigetfile("*.mat","Select External Output MAT");
        if isequal(f,0), return; end
        edtYout.Value = fullfile(pth,f);
    end

    function onCheckFcn(~,~)
        fcnName = strtrim(string(edtPredFcn.Value));
        ok = ~isempty(which(fcnName));
        if ok
            setStatus("OK: found " + fcnName);
        else
            setStatus("ERROR: function not on path: " + fcnName);
        end
    end

    function onSetSpeedIdx(~,~)
        state.speedIdx = double(edtSpeedIdx.Value);
        setStatus("SpeedIdx set to " + string(state.speedIdx));
        if ~isempty(state.Yhat), refreshAll(); end
    end

    function onSetEtaBins(~,~)
        state.etaBins = double(edtEtaBins.Value);
        setStatus("ETA bins set to " + string(state.etaBins));
        if ~isempty(state.Yhat), refreshAll(); end
    end

    function onSelectOutput(~,~)
        if isempty(state.Yhat), return; end
        updateMetricCards();
        refreshPlots();
    end

    function onNRowsChanged(~,~)
        if isempty(state.detailTable) || height(state.detailTable)==0, return; end
        n = min(round(slN.Value), height(state.detailTable));
        tblRows.Data = state.detailTable(1:n,:);
    end

    function onRun(~,~)
        try
            setStatus("Loading...");
            drawnow;

            state.modelPath  = string(strtrim(edtModel.Value));
            state.extInput   = string(strtrim(edtXin.Value));
            state.extOutput  = string(strtrim(edtYout.Value));
            state.predictFcn = string(strtrim(edtPredFcn.Value));
            state.speedIdx   = double(edtSpeedIdx.Value);
            state.etaBins    = double(edtEtaBins.Value);

            assert(isfile(state.modelPath),  "Model not found: " + state.modelPath);
            assert(isfile(state.extInput),   "Ext input not found: " + state.extInput);
            assert(isfile(state.extOutput),  "Ext output not found: " + state.extOutput);

            Xs = load(char(state.extInput));
            Ys = load(char(state.extOutput));
            state.X = double(load_matrix_any(Xs, "input"));
            state.Y = double(load_matrix_any(Ys, "output"));

            assert(size(state.Y,2) >= 5, "External output must be N-by-5");
            assert(size(state.X,1) == size(state.Y,1), "X and Y row count mismatch");

            fcnName = state.predictFcn;
            assert(~isempty(which(fcnName)), "PredictFcn not on path: " + fcnName);
            fcn = str2func(char(fcnName));

            try
                state.Yhat = fcn(state.modelPath, state.X);
            catch
                state.Yhat = fcn(state.modelPath, state.X, false);
            end
            assert(size(state.Yhat,2) >= 5, "PredictFcn must return N-by-5 (or more)");

            state.metrics = compute_metrics_table(state.Y(:,1:5), state.Yhat(:,1:5), state.outputNames);
            state.shift   = compute_shift_table(state.Y(:,1:5), state.Yhat(:,1:5), state.outputNames);
            state.detailTable = build_row_table(state.X, state.Y(:,1:5), state.Yhat(:,1:5), state.outputNames, state.speedIdx);
            state.etaBinTable = eta_bin_table(state.X, state.Y(:,1), state.Yhat(:,1), state.speedIdx, state.etaBins);

            tblMetrics.Data = state.metrics;
            tblShift.Data   = state.shift;
            tblEtaBins.Data = state.etaBinTable;

            slN.Limits = [10, max(10, size(state.X,1))];
            slN.Value = min(slN.Value, slN.Limits(2));
            onNRowsChanged();

            refreshAll();

            meanRMSE = mean(state.metrics.RMSE,'omitnan');
            etaRow = state.metrics(state.metrics.Output=="ETA",:);
            etaR2 = NaN;
            if height(etaRow)==1, etaR2 = etaRow.R2; end

            txtSummary.Value = [
                "Loaded & evaluated successfully."
                "N = " + string(size(state.X,1)) + ", Inputs = " + string(size(state.X,2))
                "PredictFcn = " + state.predictFcn
                "Mean RMSE (5 outs) = " + string(num2str(meanRMSE,'%.6g'))
                "ETA R2 = " + string(num2str(etaR2,'%.6g'))
            ];

            setStatus("Done.");
        catch ME
            setStatus("ERROR: " + string(ME.message));
            txtSummary.Value = [
                "ERROR:"
                string(ME.message)
                ""
                "Tip: check MAT fields (input/output) and predictor name on path."
            ];
            rethrow(ME);
        end
    end

    function onExportCSV(~,~)
        if isempty(state.metrics) || height(state.metrics)==0
            setStatus("Nothing to export.");
            return;
        end
        [f,pth] = uiputfile("metrics.csv","Save Metrics CSV");
        if isequal(f,0), return; end
        writetable(state.metrics, fullfile(pth,f));
        setStatus("Saved: " + string(fullfile(pth,f)));
    end

    function onExportReport(~,~)
        if isempty(state.metrics) || height(state.metrics)==0 || isempty(state.X)
            setStatus("Nothing to export (run evaluation first).");
            return;
        end

        baseDir = uigetdir(pwd, "Select export folder");
        if isequal(baseDir,0), return; end

        stamp = datestr(now,'yyyymmdd_HHMMSS');
        outDir = fullfile(baseDir, sprintf('turbo_project_report_%s', stamp));
        if ~exist(outDir,'dir'), mkdir(outDir); end

        writetable(state.metrics, fullfile(outDir, "metrics.csv"));
        writetable(state.shift,   fullfile(outDir, "shift.csv"));
        writetable(state.etaBinTable, fullfile(outDir, "eta_bins.csv"));

        nPrev = min(height(state.detailTable), 2000);
        writetable(state.detailTable(1:nPrev,:), fullfile(outDir, "rows_preview.csv"));

        pdfPath = fullfile(outDir, "report.pdf");
        spd = state.X(:, min(max(1,round(state.speedIdx)), size(state.X,2)));

        % --- Summary page (text-based)
        figS = figure('Visible','off','Color','w','Position',[100 100 1100 800]);
        ax = axes(figS); axis(ax,'off');

        lines = {};
        lines{end+1} = 'Turbo Project Report';
        lines{end+1} = ' ';
        lines{end+1} = ['Model: ' char(state.modelPath)];
        lines{end+1} = ['ExtInput: ' char(state.extInput)];
        lines{end+1} = ['ExtOutput: ' char(state.extOutput)];
        lines{end+1} = ['PredictFcn: ' char(state.predictFcn)];
        lines{end+1} = sprintf('N=%d, D=%d, SpeedIdx=%d', size(state.X,1), size(state.X,2), round(state.speedIdx));
        lines{end+1} = ['Generated: ' datestr(now)];
        lines{end+1} = ' ';
        lines{end+1} = 'Metrics (RMSE / MAE / R2):';

        for i=1:height(state.metrics)
            lines{end+1} = sprintf('  %-6s  %.6g  %.6g  %.6g', ...
                char(state.metrics.Output(i)), state.metrics.RMSE(i), state.metrics.MAE(i), state.metrics.R2(i));
        end

        text(ax, 0, 1, lines, 'VerticalAlignment','top', 'Interpreter','none', 'FontName','Consolas');
        exportgraphics(figS, pdfPath, 'ContentType','vector');
        close(figS);

        % --- Per-output pages
        for k=1:5
            nm = state.outputNames(k);
            y  = state.Y(:,k);
            yp = state.Yhat(:,k);
            res = yp - y;

            figR = figure('Visible','off','Color','w','Position',[100 100 1100 800]);
            t = tiledlayout(figR,2,2,'Padding','compact','TileSpacing','compact');

            nexttile(t);
            scatter(y, yp, '.'); grid on;
            mn=min([y;yp]); mx=max([y;yp]);
            hold on; plot([mn mx],[mn mx],'k--'); hold off;
            title([char(nm) ' Parity']); xlabel('True'); ylabel('Pred');

            nexttile(t);
            scatter(spd, res, '.'); grid on; yline(0,'k--');
            title([char(nm) ' Residual vs Speed']);
            xlabel(sprintf('Speed (col %d)', round(state.speedIdx))); ylabel('Pred - True');

            nexttile(t);
            histogram(res, 18); grid on;
            title([char(nm) ' Residual Histogram']);
            xlabel('Residual'); ylabel('Count');

            nexttile(t);
            plot(y,'-'); hold on; plot(yp,'--'); hold off; grid on;
            title([char(nm) ' True vs Pred (Row Index)']);
            xlabel('Row index'); ylabel('Value');
            legend({'True','Pred'},'Location','best');

            pngPath = fullfile(outDir, sprintf('output_%s.png', char(nm)));
            exportgraphics(figR, pngPath, 'Resolution', 200);

            exportgraphics(figR, pdfPath, 'Append', true, 'ContentType','vector');
            close(figR);
        end

        % --- ETA focus page
        figE = figure('Visible','off','Color','w','Position',[100 100 1100 800]);
        t2 = tiledlayout(figE,2,2,'Padding','compact','TileSpacing','compact');

        y  = state.Y(:,1);
        yp = state.Yhat(:,1);
        res = yp - y;

        nexttile(t2);
        bt = state.etaBinTable;
        if ~isempty(bt) && any(strcmp(bt.Properties.VariableNames,"BinCenter"))
            plot(bt.BinCenter, bt.RMSE, '-o'); grid on;
            title('ETA RMSE by Speed Bin'); xlabel('Bin center speed'); ylabel('RMSE');
        else
            axis off; text(0,0.5,'ETA bin table not available','Interpreter','none');
        end

        nexttile(t2);
        scatter(yp, res, '.'); grid on; yline(0,'k--');
        title('ETA Residual vs Pred'); xlabel('Pred ETA'); ylabel('Residual');

        nexttile(t2);
        scatter(spd, res, '.'); grid on; yline(0,'k--');
        title('ETA Residual vs Speed'); xlabel('Speed'); ylabel('Residual');

        nexttile(t2);
        scatter(y, yp, '.'); grid on;
        mn=min([y;yp]); mx=max([y;yp]);
        hold on; plot([mn mx],[mn mx],'k--'); hold off;
        title('ETA Parity'); xlabel('True'); ylabel('Pred');

        exportgraphics(figE, fullfile(outDir,'eta_focus.png'), 'Resolution', 200);
        exportgraphics(figE, pdfPath, 'Append', true, 'ContentType','vector');
        close(figE);

        setStatus("Report exported: " + string(outDir));
        txtSummary.Value = [txtSummary.Value; ""; "Exported folder:"; string(outDir)];
    end

% ===================== Rendering =====================
    function refreshAll()
        updateMetricCards();
        refreshPlots();
        refreshEtaFocus();
    end

    function updateMetricCards()
        if isempty(state.metrics) || height(state.metrics)==0
            valCardRMSE.Text = "-";
            valCardMAE.Text  = "-";
            valCardR2.Text   = "-";
            return;
        end
        outName = string(ddOut.Value);
        k = find(state.metrics.Output == outName, 1);
        if isempty(k), k = 1; end
        valCardRMSE.Text = num2str(state.metrics.RMSE(k),'%.6g');
        valCardMAE.Text  = num2str(state.metrics.MAE(k),'%.6g');
        valCardR2.Text   = num2str(state.metrics.R2(k),'%.6g');
    end

    function refreshPlots()
        outName = string(ddOut.Value);
        k = find(state.outputNames == outName, 1);
        if isempty(k), k = 1; end

        y  = state.Y(:,k);
        yp = state.Yhat(:,k);
        res = yp - y;
        spd = state.X(:, min(max(1,round(state.speedIdx)), size(state.X,2)));

        cla(axParity);
        scatter(axParity, y, yp, ".");
        grid(axParity,"on");
        mn = min([y;yp]); mx = max([y;yp]);
        hold(axParity,"on"); plot(axParity, [mn mx],[mn mx],"k--"); hold(axParity,"off");
        title(axParity, outName + " Parity");
        xlabel(axParity,"True"); ylabel(axParity,"Pred");

        cla(axResSpeed);
        scatter(axResSpeed, spd, res, ".");
        yline(axResSpeed,0,"k--");
        grid(axResSpeed,"on");
        title(axResSpeed, outName + " Residual vs Speed");
        xlabel(axResSpeed, "Speed (col " + string(round(state.speedIdx)) + ")");
        ylabel(axResSpeed, "Pred - True");

        cla(axHist);
        histogram(axHist, res, 18);
        grid(axHist,"on");
        title(axHist, outName + " Residual Histogram");
        xlabel(axHist, "Residual"); ylabel(axHist, "Count");

        cla(axSeries);
        plot(axSeries, y, "-"); hold(axSeries,"on");
        plot(axSeries, yp, "--"); hold(axSeries,"off");
        grid(axSeries,"on");
        title(axSeries, outName + " True vs Pred (Row Index)");
        xlabel(axSeries,"Row index"); ylabel(axSeries,"Value");
        legend(axSeries, {"True","Pred"}, "Location","best");
    end

    function refreshEtaFocus()
        if isempty(state.Yhat), return; end
        spd = state.X(:, min(max(1,round(state.speedIdx)), size(state.X,2)));
        y  = state.Y(:,1);
        yp = state.Yhat(:,1);
        res = yp - y;

        cla(axEtaBin);
        bt = state.etaBinTable;
        if ~isempty(bt) && any(strcmp(bt.Properties.VariableNames,"BinCenter"))
            plot(axEtaBin, bt.BinCenter, bt.RMSE, "-o");
            grid(axEtaBin,"on");
            title(axEtaBin,"ETA RMSE by Speed Bin");
            xlabel(axEtaBin,"Bin center speed"); ylabel(axEtaBin,"RMSE");
        end

        cla(axEtaResPred);
        scatter(axEtaResPred, yp, res, ".");
        yline(axEtaResPred,0,"k--");
        grid(axEtaResPred,"on");
        title(axEtaResPred,"ETA Residual vs Pred");
        xlabel(axEtaResPred,"Pred ETA"); ylabel(axEtaResPred,"Residual");

        cla(axEtaResSpeed);
        scatter(axEtaResSpeed, spd, res, ".");
        yline(axEtaResSpeed,0,"k--");
        grid(axEtaResSpeed,"on");
        title(axEtaResSpeed,"ETA Residual vs Speed");
        xlabel(axEtaResSpeed,"Speed"); ylabel(axEtaResSpeed,"Residual");

        cla(axEtaParity);
        scatter(axEtaParity, y, yp, ".");
        grid(axEtaParity,"on");
        mn = min([y;yp]); mx = max([y;yp]);
        hold(axEtaParity,"on"); plot(axEtaParity, [mn mx],[mn mx],"k--"); hold(axEtaParity,"off");
        title(axEtaParity,"ETA Parity");
        xlabel(axEtaParity,"True"); ylabel(axEtaParity,"Pred");
    end

    function setStatus(msg)
        lblStatus.Text = "Status: " + char(msg);
        drawnow;
    end
end

% ===================== Helpers =====================

function X = load_matrix_any(S, baseName)
if isfield(S, baseName)
    X = S.(baseName);
    return;
end
alts = {"X","x","input","inputs","Xin","Y","y","output","outputs","Yout"};
for i=1:numel(alts)
    nm = alts{i};
    if isfield(S, nm)
        v = S.(nm);
        if isnumeric(v) && ismatrix(v)
            X = v; return;
        end
    end
end
fns = fieldnames(S);
for i=1:numel(fns)
    v = S.(fns{i});
    if isnumeric(v) && ismatrix(v)
        X = v; return;
    end
end
error("No numeric matrix found in MAT struct.");
end

function T = compute_metrics_table(Y, Yhat, names)
names = string(names(:));
rmse = zeros(numel(names),1);
mae  = zeros(numel(names),1);
r2   = zeros(numel(names),1);

for k=1:numel(names)
    y  = Y(:,k);
    yp = Yhat(:,k);
    e  = yp - y;

    rmse(k) = sqrt(mean(e.^2,'omitnan'));
    mae(k)  = mean(abs(e),'omitnan');

    mu = mean(y,'omitnan');
    ssRes = sum((y-yp).^2,'omitnan');
    ssTot = sum((y-mu).^2,'omitnan');

    if ssTot <= eps
        r2(k) = NaN;
    else
        r2(k) = 1 - ssRes/ssTot;
    end
end

T = table(names, rmse, mae, r2, 'VariableNames', {'Output','RMSE','MAE','R2'});
end

function T = compute_shift_table(Y, Yhat, names)
names = string(names(:));
muTrue = zeros(numel(names),1);
muPred = zeros(numel(names),1);
muRes  = zeros(numel(names),1);
sdRes  = zeros(numel(names),1);

for k=1:numel(names)
    y  = Y(:,k);
    yp = Yhat(:,k);
    r = yp - y;
    muTrue(k) = mean(y,'omitnan');
    muPred(k) = mean(yp,'omitnan');
    muRes(k)  = mean(r,'omitnan');
    sdRes(k)  = std(r,'omitnan');
end

T = table(names, muTrue, muPred, muRes, sdRes, ...
    'VariableNames', {'Output','MeanTrue','MeanPred','MeanResidual','StdResidual'});
end

function T = build_row_table(X, Y, Yhat, names, speedIdx)
names = string(names(:))';
N = size(X,1);
spdCol = min(max(1, round(speedIdx)), size(X,2));
speed = X(:,spdCol);

T = table((1:N)', speed, 'VariableNames', {'Row','Speed'});

for k=1:min(5, numel(names))
    nm = names(k);
    T.(nm + "_True") = Y(:,k);
    T.(nm + "_Pred") = Yhat(:,k);
    T.(nm + "_Res")  = Yhat(:,k) - Y(:,k);
end
end

function BT = eta_bin_table(X, yEta, yhatEta, speedIdx, nBins)
spdCol = min(max(1, round(speedIdx)), size(X,2));
spd = X(:,spdCol);

q = linspace(0,1,nBins+1);
edges = unique(quantile(spd, q));
if numel(edges) < 4
    edges = linspace(min(spd), max(spd), nBins+1);
end

rmse = []; mae = []; r2 = []; binCenter = []; nIn = [];
for i=1:(numel(edges)-1)
    lo = edges(i); hi = edges(i+1);
    if i == numel(edges)-1
        idx = (spd >= lo) & (spd <= hi);
    else
        idx = (spd >= lo) & (spd < hi);
    end
    if nnz(idx) < 5, continue; end

    y = yEta(idx);
    yp = yhatEta(idx);
    e = yp - y;

    rmse(end+1,1) = sqrt(mean(e.^2,'omitnan')); %#ok<AGROW>
    mae(end+1,1)  = mean(abs(e),'omitnan'); %#ok<AGROW>

    mu = mean(y,'omitnan');
    ssRes = sum((y-yp).^2,'omitnan');
    ssTot = sum((y-mu).^2,'omitnan');
    if ssTot <= eps
        r2(end+1,1) = NaN; %#ok<AGROW>
    else
        r2(end+1,1) = 1 - ssRes/ssTot; %#ok<AGROW>
    end

    binCenter(end+1,1) = 0.5*(lo+hi); %#ok<AGROW>
    nIn(end+1,1) = nnz(idx); %#ok<AGROW>
end

BT = table(binCenter, nIn, rmse, mae, r2, ...
    'VariableNames', {'BinCenter','N','RMSE','MAE','R2'});
end