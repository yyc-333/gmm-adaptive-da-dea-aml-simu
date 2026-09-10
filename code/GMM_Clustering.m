function gmmResult = GMM_Clustering(zdata_train, zdata_test, varargin)
%% ============================================================
% GMM_Clustering
% 先基于训练集拟合 GMM，再基于后验概率/概率密度为训练集和测试集归类。
%
% 输入：
%   zdata_train : 训练集 table，第一列为 target，其余列为特征
%   zdata_test  : 测试集 table，第一列为 target，其余列为特征
%
% Name-Value 参数：
%   'MinK'      : 最小聚类数，默认 2
%   'MaxK'      : 最大聚类数，默认 4
%   'RandomSeed': 随机种子，默认 13
%   'BaseName'  : 输出文件名前缀，默认 ''
%   'OutputDir' : 若非空，则保存聚类后的 train/test CSV 与 summary
%   'MinTrainPositivePerCluster' : 每个训练 cluster 中 target=1 的最少样本数，默认 5
%% ============================================================

p = inputParser;
addRequired(p, 'zdata_train', @(x) istable(x));
addRequired(p, 'zdata_test', @(x) istable(x));
addParameter(p, 'MinK', 2, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'MaxK', 4, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'RandomSeed', 13, @(x) isnumeric(x) && isscalar(x));
addParameter(p, 'BaseName', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'OutputDir', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'MinTrainPositivePerCluster', 5, @(x) isnumeric(x) && isscalar(x) && x >= 0);
parse(p, zdata_train, zdata_test, varargin{:});

minK = max(1, round(p.Results.MinK));
maxK = max(minK, round(p.Results.MaxK));
randomSeed = p.Results.RandomSeed;
baseName = char(string(p.Results.BaseName));
outputDir = char(string(p.Results.OutputDir));
minTrainPositivePerCluster = round(p.Results.MinTrainPositivePerCluster);

rng(randomSeed, 'twister');

if width(zdata_train) < 2 || width(zdata_test) < 2
    error('GMM_Clustering:InvalidInput', '训练集和测试集必须至少包含 target 列和一个特征列。');
end

trainVarNames = zdata_train.Properties.VariableNames;
testVarNames = zdata_test.Properties.VariableNames;
if ~strcmp(trainVarNames{1}, testVarNames{1}) || ~isequal(trainVarNames(2:end), testVarNames(2:end))
    error('GMM_Clustering:VariableMismatch', '训练集和测试集的变量列必须一致。');
end

targetName = trainVarNames{1};
featureNames = trainVarNames(2:end);

y_train = table2array(zdata_train(:, 1));
y_test = table2array(zdata_test(:, 1));
X_train = table2array(zdata_train(:, 2:end));
X_test = table2array(zdata_test(:, 2:end));

X_train(isinf(X_train)) = NaN;
X_test(isinf(X_test)) = NaN;

validTrain = ~any(isnan(X_train), 2) & ~isnan(y_train);
validTest = ~any(isnan(X_test), 2) & ~isnan(y_test);

if any(~validTrain)
    fprintf('GMM: 训练集中删除含 NaN/Inf 的样本数: %d\n', sum(~validTrain));
end
if any(~validTest)
    fprintf('GMM: 测试集中删除含 NaN/Inf 的样本数: %d\n', sum(~validTest));
end

zdata_train_clean = zdata_train(validTrain, :);
zdata_test_clean = zdata_test(validTest, :);
y_train = y_train(validTrain);
y_test = y_test(validTest);
X_train = X_train(validTrain, :);
X_test = X_test(validTest, :);

var0 = var(X_train, 0, 1, 'omitnan') == 0;
if any(var0)
    fprintf('GMM: 根据训练集删除零方差变量数: %d\n', sum(var0));
    X_train(:, var0) = [];
    X_test(:, var0) = [];
    featureNames(var0) = [];
    zdata_train_clean(:, find(var0) + 1) = [];
    zdata_test_clean(:, find(var0) + 1) = [];
end

idx_train_class0 = (y_train == 0);
idx_train_class1 = (y_train == 1);
X0_train = X_train(idx_train_class0, :);
n0_train = size(X0_train, 1);
n1_train = sum(idx_train_class1);

if n0_train < minK
    error('GMM_Clustering:TooFewClass0Samples', ...
        '训练集中 target=0 样本数为 %d，不足以拟合 minK=%d 的 GMM。', n0_train, minK);
end

if minTrainPositivePerCluster > 0 && n1_train < minK * minTrainPositivePerCluster
    error('GMM_Clustering:TooFewClass1Samples', ...
        '训练集中 target=1 样本数为 %d，不足以满足 minK=%d 且每组正类不少于 %d。', ...
        n1_train, minK, minTrainPositivePerCluster);
end

maxKByPositive = maxK;
if minTrainPositivePerCluster > 0
    maxKByPositive = floor(n1_train / minTrainPositivePerCluster);
end

maxKAllowed = min([maxK, n0_train - 1, size(X0_train, 1), maxKByPositive]);
if maxKAllowed < minK
    maxKAllowed = minK;
end
kCandidates = minK:maxKAllowed;

fprintf('GMM: 仅使用训练集 target=0 样本拟合，候选聚类数 k=%s\n', mat2str(kCandidates));

BIC = inf(maxKAllowed, 1);
GMModels = cell(maxKAllowed, 1);
options = statset('MaxIter', 1000);

for k = kCandidates
    try
        GMModels{k} = fitgmdist(X0_train, k, ...
            'RegularizationValue', 1e-5, ...
            'Replicates', 10, ...
            'Options', options);
        BIC(k) = GMModels{k}.BIC;
        fprintf('  k=%d, BIC=%.4f\n', k, BIC(k));
    catch ME
        fprintf('  GMM k=%d 拟合失败: %s\n', k, ME.message);
        BIC(k) = inf;
    end
end

if all(isinf(BIC(kCandidates)))
    error('GMM_Clustering:FitFailed', '候选聚类数 2-4 的 GMM 模型均拟合失败。');
end

[~, bestKLocal] = min(BIC(kCandidates));
bestK = kCandidates(bestKLocal);
GM = GMModels{bestK};

fprintf('GMM: BIC 选择的最优聚类数: %d\n', bestK);

[train_cluster_idx, ~, train_post] = cluster(GM, X_train);

if minTrainPositivePerCluster > 0
    train_cluster_idx = enforceMinPositivePerCluster(train_cluster_idx, train_post, y_train, minTrainPositivePerCluster, bestK);
end

% 使用线性核（Linear kernel）对测试集进行归类
% K(x, mu_c) = x' * mu_c
% 测试集归类仅基于每个最终训练cluster中的 target=1 正类特征中心。
positive_cluster_centers = zeros(bestK, size(X_train, 2));
for c = 1:bestK
    idx_positive_c = (train_cluster_idx == c) & (y_train == 1);
    if any(idx_positive_c)
        positive_cluster_centers(c, :) = mean(X_train(idx_positive_c, :), 1);
    else
        idx_cluster_c = (train_cluster_idx == c);
        if any(idx_cluster_c)
            positive_cluster_centers(c, :) = mean(X_train(idx_cluster_c, :), 1);
            fprintf('GMM: cluster%d 无训练正类样本，测试归类中心回退为该cluster全部训练样本中心。\n', c);
        else
            positive_cluster_centers(c, :) = GM.mu(c, :);
            fprintf('GMM: cluster%d 无训练样本，测试归类中心回退为GMM均值。\n', c);
        end
    end
end

linear_similarity = X_test * positive_cluster_centers';

% 归一化为概率形式（softmax）
linear_similarity_shifted = linear_similarity - max(linear_similarity, [], 2);
exp_sim = exp(linear_similarity_shifted);
test_post = exp_sim ./ (sum(exp_sim, 2) + eps);

% 测试集归类：选择与训练集正类特征中心线性核相似度最大的cluster
[~, test_cluster_idx] = max(linear_similarity, [], 2);

fprintf('GMM: 测试集使用训练集正类特征中心的线性核（Linear kernel）归类完成\n');

train_density = pdf(GM, X_train);
test_density = pdf(GM, X_test);

trainCounts = accumarray(train_cluster_idx, 1, [bestK, 1], @sum, 0);
testCounts = accumarray(test_cluster_idx, 1, [bestK, 1], @sum, 0);

fprintf('GMM: 训练集各 cluster 样本数: ');
fprintf('%d ', trainCounts);
fprintf('\n');
trainPositiveCounts = accumarray(train_cluster_idx, double(y_train == 1), [bestK, 1], @sum, 0);
fprintf('GMM: 训练集各 cluster 正类样本数: ');
fprintf('%d ', trainPositiveCounts);
fprintf('\n');
fprintf('GMM: 测试集各 cluster 样本数: ');
fprintf('%d ', testCounts);
fprintf('\n');

trainTables = cell(bestK, 1);
testTables = cell(bestK, 1);
clusterSummary = cell(bestK, 10);

for c = 1:bestK
    idxTrainC = (train_cluster_idx == c);
    idxTestC = (test_cluster_idx == c);

    trainTables{c} = zdata_train_clean(idxTrainC, :);
    testTables{c} = zdata_test_clean(idxTestC, :);

    clusterSummary(c, :) = { ...
        c, ...
        sum(idxTrainC), sum(y_train(idxTrainC) == 0), sum(y_train(idxTrainC) == 1), ...
        sum(idxTestC), sum(y_test(idxTestC) == 0), sum(y_test(idxTestC) == 1), ...
        mean(train_density(idxTrainC), 'omitnan'), ...
        mean(test_density(idxTestC), 'omitnan'), ...
        bestK};
end

summaryTable = cell2table(clusterSummary, 'VariableNames', { ...
    'cluster', ...
    'n_train', 'n_train_target0', 'n_train_target1', ...
    'n_test', 'n_test_target0', 'n_test_target1', ...
    'mean_train_density', 'mean_test_density', ...
    'bestK'});

if ~isempty(outputDir)
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    if isempty(baseName)
        baseName = 'gmm_result';
    end

    writetable(summaryTable, fullfile(outputDir, [baseName, '_gmm_summary.csv']));

    posteriorTrain = array2table(train_post);
    posteriorTest = array2table(test_post);
    probNames = arrayfun(@(k) sprintf('prob_cluster%d', k), 1:bestK, 'UniformOutput', false);
    posteriorTrain.Properties.VariableNames = probNames;
    posteriorTest.Properties.VariableNames = probNames;
    posteriorTrain.cluster = train_cluster_idx;
    posteriorTrain.target = y_train;
    posteriorTrain.density = train_density;
    posteriorTest.cluster = test_cluster_idx;
    posteriorTest.target = y_test;
    posteriorTest.density = test_density;
    posteriorTrain = movevars(posteriorTrain, {'target', 'cluster', 'density'}, 'Before', 1);
    posteriorTest = movevars(posteriorTest, {'target', 'cluster', 'density'}, 'Before', 1);
    writetable(posteriorTrain, fullfile(outputDir, [baseName, '_train_posterior_prob.csv']));
    writetable(posteriorTest, fullfile(outputDir, [baseName, '_test_posterior_prob.csv']));

    for c = 1:bestK
        writetable(trainTables{c}, fullfile(outputDir, sprintf('%s_cluster%d_train.csv', baseName, c)));
        writetable(testTables{c}, fullfile(outputDir, sprintf('%s_cluster%d_test.csv', baseName, c)));
    end
end

gmmResult = struct();
gmmResult.model = GM;
gmmResult.bestK = bestK;
gmmResult.BIC = BIC;
gmmResult.kCandidates = kCandidates;
gmmResult.train_cluster_idx = train_cluster_idx;
gmmResult.test_cluster_idx = test_cluster_idx;
gmmResult.train_posterior = train_post;
gmmResult.test_posterior = test_post;
gmmResult.train_density = train_density;
gmmResult.test_density = test_density;
gmmResult.train_tables = trainTables;
gmmResult.test_tables = testTables;
gmmResult.summary = summaryTable;
gmmResult.feature_names = featureNames;
gmmResult.target_name = targetName;
gmmResult.valid_train_mask = validTrain;
gmmResult.valid_test_mask = validTest;
gmmResult.min_train_positive_per_cluster = minTrainPositivePerCluster;
end

function cluster_idx = enforceMinPositivePerCluster(cluster_idx, posterior, y, minPositive, nClusters)
positiveIdx = find(y == 1);
positiveCounts = accumarray(cluster_idx, double(y == 1), [nClusters, 1], @sum, 0);

if sum(positiveCounts) < minPositive * nClusters
    error('GMM_Clustering:TooFewClass1Samples', ...
        '训练集正类总数为 %d，不足以保证 %d 个 cluster 每组至少 %d 个正类样本。', ...
        sum(positiveCounts), nClusters, minPositive);
end

maxIter = numel(positiveIdx) * nClusters;
iter = 0;

while any(positiveCounts < minPositive)
    iter = iter + 1;
    if iter > maxIter
        error('GMM_Clustering:PositiveBalanceFailed', ...
            '无法通过重分配正类样本满足每组正类不少于 %d 的约束。', minPositive);
    end

    deficientClusters = find(positiveCounts < minPositive);
    targetCluster = deficientClusters(1);
    donorClusters = find(positiveCounts > minPositive);

    bestSample = [];
    bestDonor = [];
    bestScore = -inf;

    for d = donorClusters(:)'
        donorPositiveIdx = positiveIdx(cluster_idx(positiveIdx) == d);
        if isempty(donorPositiveIdx)
            continue;
        end

        [candidateScore, localIdx] = max(posterior(donorPositiveIdx, targetCluster));
        if candidateScore > bestScore
            bestScore = candidateScore;
            bestSample = donorPositiveIdx(localIdx);
            bestDonor = d;
        end
    end

    if isempty(bestSample)
        error('GMM_Clustering:PositiveBalanceFailed', ...
            'cluster%d 正类不足，但找不到可转移的正类样本。', targetCluster);
    end

    cluster_idx(bestSample) = targetCluster;
    positiveCounts(bestDonor) = positiveCounts(bestDonor) - 1;
    positiveCounts(targetCluster) = positiveCounts(targetCluster) + 1;
end
end
