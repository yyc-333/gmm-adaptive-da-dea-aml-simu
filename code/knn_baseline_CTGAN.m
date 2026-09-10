clc; clear; close all;

%% =====================================================
%   KNN baseline for CTGAN simulation files
%   Modified version:
%   1) Keep the two-layer micro summary logic of rf_baseline_CTGAN_two_layer_micro.m
%   2) Add correlation-based feature groups: best_group
%   3) Use KNN + sample Weights, no Cost matrix
%   4) Do NOT add the all-feature group; groups are only correlation-based
%   5) Select the F1-optimal threshold on the training set and use it for prediction
%   6) Jointly select best_group + best KNN parameters by comprehensive score
%      score = weighted harmonic mean of F1/AUC/Recall/Precision/Accuracy
% =====================================================

%% 路径设置
dataFolder = 'D:\白血病模拟数据\Simulated_data_MC_3_3000';
resultRootFolder = 'D:\白血病模拟数据\Simulated_data_MC_3_3000\DADEA_CTGAN_100_result';
resultFolder = fullfile(resultRootFolder, 'KNN_baseline_bestgroup_knn_result');

if ~exist(resultFolder, 'dir')
    mkdir(resultFolder);
end

%% 参数设置
randomSeed = 9;
rng(randomSeed, 'twister');
fprintf('随机数种子已设置为: %d\n', randomSeed);

K_fold_requested = 3;
correlationThreshold = 0.9;
useSampleWeights = true;  % KNN样本权重；不使用Cost代价矩阵

kGrid = [3, 5, 7, 9, 15, 25];
% KNN邻居数候选参数

files = dir(fullfile(dataFolder, 'synthetic_CTGAN_*.csv'));
fprintf('发现 %d 个 CSV 文件，开始 KNN-best_group 对照实验...\n', length(files));

oldPatterns = { ...
    'synthetic_CTGAN_*_global_micro_metrics.csv', ...
    'synthetic_CTGAN_*_cluster*_metrics.csv', ...
    'best_summary.csv', ...
    'global_micro_metrics_arithmetic_mean.csv'};
for pi = 1:numel(oldPatterns)
    oldFiles = dir(fullfile(resultFolder, oldPatterns{pi}));
    for oi = 1:numel(oldFiles)
        delete(fullfile(oldFiles(oi).folder, oldFiles(oi).name));
    end
end

summaryFile = fullfile(resultFolder, 'best_summary.csv');
fid = fopen(summaryFile, 'w', 'n', 'GBK');
fprintf(fid, ['filename,best_group,num_group_features,best_num_trees,best_min_leaf,best_mtry,best_threshold,best_CV_score,best_CV_F1,best_CV_AUC,' ...
    'best_CV_precision,best_CV_recall,best_CV_accuracy,best_CV_TP,best_CV_FP,best_CV_FN,best_CV_TN,' ...
    'best_train_F1,best_train_AUC,best_train_precision,best_train_recall,best_train_accuracy,best_train_TP,best_train_FP,best_train_FN,best_train_TN,' ...
    'best_test_F1,best_test_AUC,best_test_precision,best_test_recall,best_test_accuracy,best_test_TP,best_test_FP,best_test_FN,best_test_TN,best_group_features\n']);
fclose(fid);

%% =====================================================
%   批量处理每个 synthetic_CTGAN 文件
%% =====================================================
for fi = 1:length(files)
    originalFileName = files(fi).name;
    originalBaseName = erase(originalFileName, '.csv');
    filePath = fullfile(dataFolder, originalFileName);

    fprintf('\n===============================\n');
    fprintf('正在处理文件：%s\n', originalFileName);
    fprintf('===============================\n');

    dataTable = readtable(filePath, 'VariableNamingRule', 'preserve');
    y_all = table2array(dataTable(:, 1));

    idx0 = find(y_all == 0);
    idx1 = find(y_all == 1);
    n0 = numel(idx0);
    n1 = numel(idx1);

    if n0 < 2 || n1 < 2
        fprintf('跳过 %s：类别样本不足。target0=%d, target1=%d\n', originalFileName, n0, n1);
        continue;
    end

    %% 60%/40% 分层划分
    n_train0 = round(n0 * 0.70);
    n_train1 = round(n1 * 0.70);
    n_test0 = max(n0 - n_train0, 1);
    n_test1 = max(n1 - n_train1, 1);
    n_train0 = n0 - n_test0;
    n_train1 = n1 - n_test1;

    idx0 = idx0(randperm(n0));
    idx1 = idx1(randperm(n1));
    train_idx = sort([idx0(1:n_train0); idx1(1:n_train1)]);
    test_idx = sort([idx0(n_train0+1:end); idx1(n_train1+1:end)]);

    trainTableAll = dataTable(train_idx, :);
    testTableAll = dataTable(test_idx, :);

    fprintf('数据划分完成：训练集 %d，测试集 %d\n', height(trainTableAll), height(testTableAll));

    %% GMM 聚类：优先调用 GMM_Clustering；否则作为单 cluster 处理
    gmmFolder = fullfile(resultFolder, 'GMM_after_split');
    if exist('GMM_Clustering', 'file') == 2
        gmmResult = GMM_Clustering(trainTableAll, testTableAll, ...
            'MinK', 3, ...
            'MaxK', 6, ...
            'MinTrainPositivePerCluster', 12, ...
            'RandomSeed', randomSeed, ...
            'BaseName', originalBaseName, ...
            'OutputDir', gmmFolder);
        nClusters = gmmResult.bestK;
        trainTables = gmmResult.train_tables;
        testTables = gmmResult.test_tables;
    else
        fprintf('提示：未找到 GMM_Clustering，当前文件作为单一 cluster 处理。\n');
        nClusters = 1;
        trainTables = {trainTableAll};
        testTables = {testTableAll};
    end

    file_TP = [];
    file_FP = [];
    file_FN = [];
    file_TN = [];
    file_scores = [];
    file_labels = [];

    file_train_TP = [];
    file_train_FP = [];
    file_train_FN = [];
    file_train_TN = [];
    file_train_scores = [];
    file_train_labels = [];

    file_cv_TP = [];
    file_cv_FP = [];
    file_cv_FN = [];
    file_cv_TN = [];
    file_cv_scores = [];
    file_cv_labels = [];

    for ci = 1:nClusters
        trainTable = trainTables{ci};
        testTable = testTables{ci};
        clusterName = sprintf('%s_cluster%d', originalBaseName, ci);

        if height(trainTable) < 3 || height(testTable) < 1
            fprintf('跳过 %s：训练集或测试集样本不足。\n', clusterName);
            continue;
        end

        y_train = double(table2array(trainTable(:, 1)));
        y_test = double(table2array(testTable(:, 1)));

        n_train0_c = sum(y_train == 0);
        n_train1_c = sum(y_train == 1);
        minClass = min(n_train0_c, n_train1_c);
        if numel(unique(y_train)) < 2 || minClass < 2
            fprintf('跳过 %s：训练集类别不足。target0=%d, target1=%d\n', clusterName, n_train0_c, n_train1_c);
            continue;
        end

        X_train_full = table2array(trainTable(:, 2:end));
        X_test_full = table2array(testTable(:, 2:end));
        varNames = trainTable.Properties.VariableNames(2:end);

        K_fold = min(K_fold_requested, minClass);
        if K_fold < 2
            fprintf('跳过 %s：无法进行分层交叉验证。\n', clusterName);
            continue;
        end

        fprintf('\n开始 KNN-KNN：%s | train=%d, test=%d, K=%d\n', ...
            clusterName, height(trainTable), height(testTable), K_fold);

        %% 仅基于训练集生成相关性候选 group
        groups_unique = make_correlation_groups(X_train_full, correlationThreshold);
        nGroups = numel(groups_unique);
        groupSizes = cellfun(@numel, groups_unique);
        fprintf('特征分组完成：共 %d 个候选指标组合；group大小范围=[%d,%d]\n', ...
            nGroups, min(groupSizes), max(groupSizes));

        %% 分层 K 折索引
        idx_train0 = find(y_train == 0);
        idx_train1 = find(y_train == 1);
        idx_train0 = idx_train0(randperm(numel(idx_train0)));
        idx_train1 = idx_train1(randperm(numel(idx_train1)));
        cv0 = make_stratified_folds(idx_train0, K_fold);
        cv1 = make_stratified_folds(idx_train1, K_fold);

        bestScore = -Inf;
        bestParams = struct();
        bestCV = struct();
        bestGroup = NaN;
        bestGroupCols = [];

        %% =====================================================
        %   联合遍历：best_group + KNN KNN参数
        %% =====================================================
        for gi = 1:nGroups
            cols = groups_unique{gi};
            X_group = X_train_full(:, cols);
            p_group = numel(cols);
            dummyLeafGrid = 1;
            dummyMtryGrid = 1;

            for ti = 1:numel(kGrid)
                for li = 1
                    for mi = 1
                        cv_TP = zeros(K_fold, 1);
                        cv_FP = zeros(K_fold, 1);
                        cv_FN = zeros(K_fold, 1);
                        cv_TN = zeros(K_fold, 1);
                        cv_scores_all = [];
                        cv_labels_all = [];
                        cv_thresholds = [];

                        for k = 1:K_fold
                            val_idx = sort([cv0{k}; cv1{k}]);
                            fit_idx = setdiff((1:numel(y_train))', val_idx);

                            if numel(unique(y_train(fit_idx))) < 2 || numel(unique(y_train(val_idx))) < 2
                                cv_TP(k) = NaN;
                                cv_FP(k) = NaN;
                                cv_FN(k) = NaN;
                                cv_TN(k) = NaN;
                                continue;
                            end

                            try
                                model = fit_knn_rf(X_group(fit_idx, :), y_train(fit_idx), ...
                                    kGrid(ti), useSampleWeights);
                            catch ME
                                fprintf('  警告：%s group%d fold%d KNN拟合失败：%s\n', ...
                                    clusterName, gi, k, ME.message);
                                cv_TP(k) = NaN;
                                cv_FP(k) = NaN;
                                cv_FN(k) = NaN;
                                cv_TN(k) = NaN;
                                continue;
                            end

                            [~, score_val] = predict_knn_positive_score(model, X_group(val_idx, :));
                            y_val = y_train(val_idx);
                            pred_val = double(score_val >= model.threshold);

                            [TP, FP, FN, TN] = confusion_counts(y_val, pred_val);
                            cv_TP(k) = TP;
                            cv_FP(k) = FP;
                            cv_FN(k) = FN;
                            cv_TN(k) = TN;
                            cv_scores_all = [cv_scores_all; score_val(:)];
                            cv_labels_all = [cv_labels_all; double(y_val(:) == 1)];
                            cv_thresholds = [cv_thresholds; model.threshold];
                        end

                        validFold = ~isnan(cv_TP);
                        if ~any(validFold)
                            continue;
                        end

                        TP_total = sum(cv_TP(validFold));
                        FP_total = sum(cv_FP(validFold));
                        FN_total = sum(cv_FN(validFold));
                        TN_total = sum(cv_TN(validFold));
                        [precision_cv, recall_cv, F1_cv, accuracy_cv] = metrics_from_counts(TP_total, FP_total, FN_total, TN_total);
                        AUC_cv = rank_auc(cv_scores_all, cv_labels_all);

                        metricVector = [F1_cv, AUC_cv, recall_cv, precision_cv, accuracy_cv];
                        metricVector(~isfinite(metricVector)) = 0;
                        weights = [0.30, 0.25, 0.25, 0.10, 0.10];
                        score = sum(weights) / sum(weights ./ max(metricVector, eps));

                        if score > bestScore
                            bestScore = score;
                            bestGroup = gi;
                            bestGroupCols = cols;
                            bestParams.k = kGrid(ti);
                            bestParams.dummyLeaf = 1;
                            bestParams.dummyMtry = 1;
                            bestCV.F1 = F1_cv;
                            bestCV.AUC = AUC_cv;
                            bestCV.precision = precision_cv;
                            bestCV.recall = recall_cv;
                            bestCV.accuracy = accuracy_cv;
                            bestCV.TP = TP_total;
                            bestCV.FP = FP_total;
                            bestCV.FN = FN_total;
                            bestCV.TN = TN_total;
                            bestCV.scores = cv_scores_all;
                            bestCV.labels = cv_labels_all;
                            bestCV.threshold = median(cv_thresholds, 'omitnan');
                        end
                    end
                end
            end
        end

        if bestScore == -Inf || isempty(bestGroupCols)
            fprintf('跳过 %s：所有 best_group + KNN 参数组合均不可用。\n', clusterName);
            continue;
        end

        %% 使用 best_group + best KNN 参数在完整训练集重训
        X_train_best = X_train_full(:, bestGroupCols);
        X_test_best = X_test_full(:, bestGroupCols);

        finalModel = fit_knn_rf(X_train_best, y_train, ...
            bestParams.k, useSampleWeights);

        [~, score_train] = predict_knn_positive_score(finalModel, X_train_best);
        pred_train = double(score_train >= finalModel.threshold);
        [TP_train, FP_train, FN_train, TN_train] = confusion_counts(y_train, pred_train);
        [precision_train, recall_train, F1_train, accuracy_train] = metrics_from_counts(TP_train, FP_train, FN_train, TN_train);
        AUC_train = rank_auc(score_train, double(y_train == 1));

        [~, score_test] = predict_knn_positive_score(finalModel, X_test_best);
        pred_test = double(score_test >= finalModel.threshold);
        [TP_test, FP_test, FN_test, TN_test] = confusion_counts(y_test, pred_test);
        [precision_test, recall_test, F1_test, accuracy_test] = metrics_from_counts(TP_test, FP_test, FN_test, TN_test);
        AUC_test = rank_auc(score_test, double(y_test == 1));

        fprintf('KNN-CV micro：group=%d, nFeat=%d, Trees=%d, Leaf=%d, mtry=%d, CV_Thr=%.6f, Final_Thr=%.6f, Score=%.4f\n', ...
            bestGroup, numel(bestGroupCols), bestParams.k, bestParams.dummyLeaf, bestParams.dummyMtry, bestCV.threshold, finalModel.threshold, bestScore);
        fprintf('KNN-CV micro：F1=%.4f, AUC=%.4f, Precision=%.4f, Recall=%.4f, Accuracy=%.4f\n', ...
            bestCV.F1, bestCV.AUC, bestCV.precision, bestCV.recall, bestCV.accuracy);
        fprintf('KNN训练集性能：F1=%.4f, AUC=%.4f, Precision=%.4f, Recall=%.4f, Accuracy=%.4f\n', ...
            F1_train, AUC_train, precision_train, recall_train, accuracy_train);
        fprintf('KNN测试集性能：F1=%.4f, AUC=%.4f, Precision=%.4f, Recall=%.4f, Accuracy=%.4f\n', ...
            F1_test, AUC_test, precision_test, recall_test, accuracy_test);

        file_cv_TP(end+1) = bestCV.TP;
        file_cv_FP(end+1) = bestCV.FP;
        file_cv_FN(end+1) = bestCV.FN;
        file_cv_TN(end+1) = bestCV.TN;
        file_cv_scores = [file_cv_scores; bestCV.scores(:)];
        file_cv_labels = [file_cv_labels; bestCV.labels(:)];

        file_train_TP(end+1) = TP_train;
        file_train_FP(end+1) = FP_train;
        file_train_FN(end+1) = FN_train;
        file_train_TN(end+1) = TN_train;
        file_train_scores = [file_train_scores; score_train(:)];
        file_train_labels = [file_train_labels; double(y_train(:) == 1)];

        file_TP(end+1) = TP_test;
        file_FP(end+1) = FP_test;
        file_FN(end+1) = FN_test;
        file_TN(end+1) = TN_test;
        file_scores = [file_scores; score_test(:)];
        file_labels = [file_labels; double(y_test(:) == 1)];

        bestGroupFeatureNames = varNames(bestGroupCols);
        bestGroupFeaturesText = strjoin(cellstr(string(bestGroupFeatureNames)), '|');

        clusterFile = fullfile(resultFolder, [clusterName, '_metrics.csv']);
        write_metric_file(clusterFile, { ...
            'best_group', bestGroup; 'num_group_features', numel(bestGroupCols); 'best_num_trees', bestParams.k; 'best_min_leaf', bestParams.dummyLeaf; 'best_mtry', bestParams.dummyMtry; 'best_threshold', finalModel.threshold; 'best_CV_score', bestScore; ...
            'best_CV_F1', bestCV.F1; 'best_CV_AUC', bestCV.AUC; 'best_CV_precision', bestCV.precision; 'best_CV_recall', bestCV.recall; 'best_CV_accuracy', bestCV.accuracy; ...
            'best_CV_TP', bestCV.TP; 'best_CV_FP', bestCV.FP; 'best_CV_FN', bestCV.FN; 'best_CV_TN', bestCV.TN; ...
            'best_train_F1', F1_train; 'best_train_AUC', AUC_train; 'best_train_precision', precision_train; 'best_train_recall', recall_train; 'best_train_accuracy', accuracy_train; ...
            'best_train_TP', TP_train; 'best_train_FP', FP_train; 'best_train_FN', FN_train; 'best_train_TN', TN_train; ...
            'best_test_F1', F1_test; 'best_test_AUC', AUC_test; 'best_test_precision', precision_test; 'best_test_recall', recall_test; 'best_test_accuracy', accuracy_test; ...
            'best_test_TP', TP_test; 'best_test_FP', FP_test; 'best_test_FN', FN_test; 'best_test_TN', TN_test; ...
            'best_group_features', bestGroupFeaturesText});

        fid = fopen(summaryFile, 'a', 'n', 'GBK');
        fprintf(fid, '%s,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d,%d,%d,%s\n', ...
            clusterName, bestGroup, numel(bestGroupCols), bestParams.k, bestParams.dummyLeaf, bestParams.dummyMtry, finalModel.threshold, bestScore, ...
            bestCV.F1, bestCV.AUC, bestCV.precision, bestCV.recall, bestCV.accuracy, bestCV.TP, bestCV.FP, bestCV.FN, bestCV.TN, ...
            F1_train, AUC_train, precision_train, recall_train, accuracy_train, TP_train, FP_train, FN_train, TN_train, ...
            F1_test, AUC_test, precision_test, recall_test, accuracy_test, TP_test, FP_test, FN_test, TN_test, bestGroupFeaturesText);
        fclose(fid);

        save(fullfile(resultFolder, [clusterName, '_KNN_bestgroup.mat']), ...
            'finalModel', 'bestGroup', 'bestGroupCols', 'bestParams', 'bestCV', ...
            'bestScore', 'bestGroupFeatureNames', 'X_train_best', 'X_test_best', 'y_train', 'y_test');
    end

    %% 当前 synthetic_CTGAN 文件的 micro-level 汇总
    if ~isempty(file_TP)
        [cv_metrics, cv_counts] = aggregate_metrics(file_cv_TP, file_cv_FP, file_cv_FN, file_cv_TN, file_cv_scores, file_cv_labels);
        [train_metrics, train_counts] = aggregate_metrics(file_train_TP, file_train_FP, file_train_FN, file_train_TN, file_train_scores, file_train_labels);
        [test_metrics, test_counts] = aggregate_metrics(file_TP, file_FP, file_FN, file_TN, file_scores, file_labels);

        globalFile = fullfile(resultFolder, [originalBaseName, '_global_micro_metrics.csv']);
        write_metric_file(globalFile, { ...
            'best_CV_F1', cv_metrics.F1; 'best_CV_AUC', cv_metrics.AUC; 'best_CV_precision', cv_metrics.precision; 'best_CV_recall', cv_metrics.recall; 'best_CV_accuracy', cv_metrics.accuracy; ...
            'best_CV_TP', cv_counts.TP; 'best_CV_FP', cv_counts.FP; 'best_CV_FN', cv_counts.FN; 'best_CV_TN', cv_counts.TN; ...
            'best_train_F1', train_metrics.F1; 'best_train_AUC', train_metrics.AUC; 'best_train_precision', train_metrics.precision; 'best_train_recall', train_metrics.recall; 'best_train_accuracy', train_metrics.accuracy; ...
            'best_train_TP', train_counts.TP; 'best_train_FP', train_counts.FP; 'best_train_FN', train_counts.FN; 'best_train_TN', train_counts.TN; ...
            'Precision', test_metrics.precision; 'Recall', test_metrics.recall; 'F1', test_metrics.F1; 'Accuracy', test_metrics.accuracy; 'AUC', test_metrics.AUC; ...
            'TP', test_counts.TP; 'FP', test_counts.FP; 'FN', test_counts.FN; 'TN', test_counts.TN});
        fprintf('当前文件KNN micro指标已保存到：%s\n', globalFile);
    end
end

%% =====================================================
%   跨 synthetic_CTGAN 文件算术平均和标准差
%% =====================================================
metricFiles = dir(fullfile(resultFolder, 'synthetic_CTGAN_*_global_micro_metrics.csv'));
metricNames = {'best_CV_F1', 'best_CV_AUC', 'best_CV_precision', 'best_CV_recall', 'best_CV_accuracy', ...
               'best_train_F1', 'best_train_AUC', 'best_train_precision', 'best_train_recall', 'best_train_accuracy', ...
               'Precision', 'Recall', 'F1', 'Accuracy', 'AUC'};

if ~isempty(metricFiles)
    metricValues = NaN(numel(metricFiles), numel(metricNames));
    for mf = 1:numel(metricFiles)
        metricTable = readtable(fullfile(metricFiles(mf).folder, metricFiles(mf).name), 'VariableNamingRule', 'preserve');
        for mi = 1:numel(metricNames)
            rowIdx = strcmp(string(metricTable.Metric), metricNames{mi});
            if any(rowIdx)
                metricValues(mf, mi) = double(metricTable.Value(find(rowIdx, 1)));
            end
        end
    end

    outputTable = table( ...
        string(metricNames(:)), ...
        mean(metricValues, 1, 'omitnan')', ...
        std(metricValues, 0, 1, 'omitnan')', ...
        sum(~isnan(metricValues), 1)', ...
        'VariableNames', {'Metric', 'ArithmeticMean', 'Std', 'NumFiles'});
    outputFile = fullfile(resultFolder, 'global_micro_metrics_arithmetic_mean.csv');
    writetable(outputTable, outputFile, 'Encoding', 'GBK');
    fprintf('KNN跨文件算术平均和标准差已保存到：%s\n', outputFile);
else
    fprintf('未找到 synthetic_CTGAN_*_global_micro_metrics.csv，跳过跨文件汇总。\n');
end

fprintf('\nKNN-best_group 对照实验完成。结果目录：%s\n', resultFolder);

%% =====================================================
%   Local functions
%% =====================================================

function groups_unique = make_correlation_groups(X_train, threshold)
    X_train = double(X_train);
    if isempty(X_train)
        groups_unique = {};
        return;
    end

    [R, ~] = corrcoef(X_train, 'Rows', 'pairwise');
    R(~isfinite(R)) = 0;

    numVars = size(X_train, 2);
    groups = {};

    for i = 1:numVars
        newGroup = i;
        for j = 1:numVars
            if j == i
                continue;
            end
            corrWithGroup = abs(R(j, newGroup));
            if all(corrWithGroup <= threshold)
                newGroup = [newGroup, j]; %#ok<AGROW>
            end
        end
        groups{end+1} = sort(unique(newGroup), 'ascend'); %#ok<AGROW>
    end


    groupStrings = cellfun(@(x) sprintf('%d_', x), groups, 'UniformOutput', false);
    [~, uniqueIdx] = unique(groupStrings, 'stable');
    groups_unique = groups(uniqueIdx);
end

function folds = make_stratified_folds(indices, K)
    folds = cell(K, 1);
    n = numel(indices);
    foldSize = floor(n / K);
    for k = 1:K
        if k < K
            folds{k} = indices((k-1)*foldSize+1:k*foldSize);
        else
            folds{k} = indices((k-1)*foldSize+1:end);
        end
    end
end

function model = fit_knn_rf(X, y, k, useSampleWeights)
    % KNN分类器：保持原RF流程（标准化、训练阈值优化、正类score输出）
    y = double(y(:));
    X = double(X);

    model.mu = mean(X, 1, 'omitnan');
    model.sigma = std(X, 0, 1, 'omitnan');
    model.sigma(model.sigma == 0 | isnan(model.sigma) | ~isfinite(model.sigma)) = 1;

    Xz = (X - model.mu) ./ model.sigma;
    Xz(~isfinite(Xz)) = 0;

    if useSampleWeights
        % KNN通过距离权重间接处理类别不平衡
        model.knnModel = fitcknn(Xz, y, ...
            'NumNeighbors', k, ...
            'Distance', 'euclidean', ...
            'DistanceWeight', 'inverse', ...
            'Standardize', false);
    else
        model.knnModel = fitcknn(Xz, y, ...
            'NumNeighbors', k, ...
            'Distance', 'euclidean', ...
            'DistanceWeight', 'equal', ...
            'Standardize', false);
    end

    model.k = k;

    % 使用训练集概率选择F1最优阈值
    [~, score_train] = predict(model.knnModel, Xz);
    if size(score_train,2) == 1
        score_train = score_train(:,1);
    else
        score_train = score_train(:,2);
    end
    score_train = normalize_score_if_needed(score_train);
    model.threshold = choose_f1_threshold(y, score_train);
end


function [pred, scorePos] = predict_knn_positive_score(model, X)
    X = double(X);
    Xz = (X - model.mu) ./ model.sigma;
    Xz(~isfinite(Xz)) = 0;

    [pred_raw, score] = predict(model.knnModel, Xz);
    pred = double(pred_raw);

    if size(score,2) == 1
        scorePos = score(:,1);
    else
        scorePos = score(:,2);
    end
    scorePos = normalize_score_if_needed(scorePos);
end



function pred = convert_pred_to_double(pred_raw)
    if iscell(pred_raw)
        pred = str2double(pred_raw);
    elseif iscategorical(pred_raw)
        pred = str2double(string(pred_raw));
    elseif isstring(pred_raw)
        pred = str2double(pred_raw);
    else
        pred = double(pred_raw);
    end
    pred = pred(:);
    pred(~isfinite(pred)) = 0;
end

function score_out = normalize_score_if_needed(score_in)
    score_out = double(score_in(:));
    if isempty(score_out)
        return;
    end

    finiteIdx = isfinite(score_out);
    if ~any(finiteIdx)
        score_out(:) = 0.5;
        return;
    end

    minScore = min(score_out(finiteIdx));
    maxScore = max(score_out(finiteIdx));

    if minScore < 0 || maxScore > 1
        if maxScore > minScore
            score_out(finiteIdx) = (score_out(finiteIdx) - minScore) / (maxScore - minScore);
        else
            score_out(finiteIdx) = 0.5;
        end
    end

    score_out(~finiteIdx) = 0.5;
    score_out = max(min(score_out, 1), 0);
end


function threshold = choose_f1_threshold(y_true, scores)
    y_true = double(y_true(:));
    scores = double(scores(:));

    finiteIdx = isfinite(scores);
    y_true = y_true(finiteIdx);
    scores = scores(finiteIdx);

    if isempty(scores)
        threshold = 0.5;
        return;
    end

    thresholds = unique(scores);
    thresholds = sort(thresholds, 'descend');

    bestF1 = -Inf;
    threshold = 0.5;

    for i = 1:numel(thresholds)
        y_pred = double(scores >= thresholds(i));
        TP = sum(y_true == 1 & y_pred == 1);
        FP = sum(y_true == 0 & y_pred == 1);
        FN = sum(y_true == 1 & y_pred == 0);
        TN = sum(y_true == 0 & y_pred == 0);
        [~, ~, F1, ~] = metrics_from_counts(TP, FP, FN, TN);

        if isfinite(F1) && F1 > bestF1
            bestF1 = F1;
            threshold = thresholds(i);
        end
    end
end

function [TP, FP, FN, TN] = confusion_counts(yTrue, yPred)
    yTrue = double(yTrue(:));
    yPred = double(yPred(:));
    TP = sum(yTrue == 1 & yPred == 1);
    FP = sum(yTrue == 0 & yPred == 1);
    FN = sum(yTrue == 1 & yPred == 0);
    TN = sum(yTrue == 0 & yPred == 0);
end

function [precision, recall, F1, accuracy] = metrics_from_counts(TP, FP, FN, TN)
    precision = TP / (TP + FP + eps);
    recall = TP / (TP + FN + eps);
    F1 = 2 * precision * recall / (precision + recall + eps);
    accuracy = (TP + TN) / (TP + FP + FN + TN + eps);
end

function AUC = rank_auc(scores, labels)
    scores = double(scores(:));
    labels = double(labels(:));
    pos = labels == 1;
    neg = labels == 0;
    if sum(pos) > 0 && sum(neg) > 0
        ranks = tiedrank(scores);
        m = sum(pos);
        n = sum(neg);
        S = sum(ranks(pos));
        AUC = (S - m*(m+1)/2) / (m*n + eps);
    else
        AUC = NaN;
    end
end

function [metrics, counts] = aggregate_metrics(TP_list, FP_list, FN_list, TN_list, scores, labels)
    counts.TP = sum(TP_list);
    counts.FP = sum(FP_list);
    counts.FN = sum(FN_list);
    counts.TN = sum(TN_list);
    [metrics.precision, metrics.recall, metrics.F1, metrics.accuracy] = ...
        metrics_from_counts(counts.TP, counts.FP, counts.FN, counts.TN);
    metrics.AUC = rank_auc(scores, labels);
end

function write_metric_file(filePath, rows)
    fid = fopen(filePath, 'w', 'n', 'GBK');
    if fid == -1
        fprintf('警告：无法写入文件：%s\n', filePath);
        return;
    end
    fprintf(fid, 'Metric,Value\n');
    for i = 1:size(rows, 1)
        key = rows{i, 1};
        value = rows{i, 2};
        if isnumeric(value)
            if isfinite(value) && abs(value - round(value)) < 1e-12 && any(contains(key, {'TP', 'FP', 'FN', 'TN', 'group', 'num_', 'trees', 'leaf', 'mtry'}))
                fprintf(fid, '%s,%d\n', key, round(value));
            else
                fprintf(fid, '%s,%.6f\n', key, value);
            end
        else
            fprintf(fid, '%s,%s\n', key, string(value));
        end
    end
    fclose(fid);
end
