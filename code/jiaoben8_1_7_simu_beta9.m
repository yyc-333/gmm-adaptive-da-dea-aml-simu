clc; clear; close all;

%% 路径设置
dataFolder = 'D:\白血病模拟数据\Simulated_data_MC_1_5000';   % Python生成的100个CTGAN模拟CSV所在文件夹
resultRootFolder = 'D:\白血病模拟数据\Simulated_data_MC_1_5000\DADEA_CTGAN_100_result';    % 原始结果根目录
resultFolder = fullfile(resultRootFolder, 'bestCV_train_micro_mean_beta10_result');    % 本次修改版独立输出文件夹，避免覆盖原始结果

%% ============================================
%   运行模式设置
% ============================================
% false：完整运行（模型计算 + 结果输出 + 可视化）
% true ：仅运行可视化（要求结果文件夹中已经存在 *_global_micro_metrics.csv）
runVisualizationOnly = false;

% beta位点数量在两种模式下都需要，用于读取已有结果并绘图
nBetaRequested = 9;

% 统一放大图片文字字号
FONT_TICK   = 18;   % 坐标轴刻度
FONT_LABEL  = 20;   % x/y轴标签
FONT_TITLE  = 22;   % 标题
FONT_LEGEND = 17;   % 图例
FONT_TEXT   = 17;   % 图中注释/数值

set(0, 'DefaultAxesFontSize', FONT_TICK);
set(0, 'DefaultTextFontSize', FONT_TEXT);
set(0, 'DefaultLegendFontSize', FONT_LEGEND);

if runVisualizationOnly
    fprintf('\n============================================\n');
    fprintf('当前模式：仅运行可视化，不重新进行模型计算。\n');
    fprintf('结果目录：%s\n', resultFolder);
    fprintf('============================================\n\n');
end

if ~exist(resultFolder, 'dir')
    mkdir(resultFolder);
end

if ~runVisualizationOnly

disabledOutputFiles = { ...
    'all_files_CV_AUC_bar_with_std.png', ...
    'all_files_CV_F1_bar_with_std.png', ...
    'all_files_test_PR_combined.png', ...
    'all_files_test_ROC_combined.png', ...
    'global_micro_metrics.csv', ...
    'global_micro_metrics_arithmetic_mean.csv', ...
    'beta_position_F1_recall_performance.png'};

for i_cleanup = 1:numel(disabledOutputFiles)
    disabledPath = fullfile(resultFolder, disabledOutputFiles{i_cleanup});
    if exist(disabledPath, 'file')
        delete(disabledPath);
    end
end

disabledOutputPatterns = { ...
    '*_score_density_R1R2.png', ...
    '*_PCA_cluster.png', ...
    '*_theta_radar_test.png', ...
    '*_variables_weights_gbk.csv', ...
    'synthetic_CTGAN_*_global_micro_metrics.csv', ...
    'synthetic_CTGAN_*_cluster*_metrics.csv'};

for i_cleanup = 1:numel(disabledOutputPatterns)
    disabledFiles = dir(fullfile(resultFolder, disabledOutputPatterns{i_cleanup}));
    for j_cleanup = 1:numel(disabledFiles)
        delete(fullfile(disabledFiles(j_cleanup).folder, disabledFiles(j_cleanup).name));
    end
end

%% ============================================
%   设置随机数种子，确保数据集划分结果可重复
% ============================================
randomSeed = 9;  % 可修改此值以改变随机划分结果
rng(randomSeed, 'twister');  % 使用Mersenne Twister生成器
fprintf('随机数种子已设置为: %d\n', randomSeed);

%% ============================================
%   交叉验证设置
% ============================================
K_fold = 3;  % K折交叉验证的折数，可修改
K_fold_requested = K_fold;  % 每个GMM cluster会根据少数类样本数自适应调整实际折数
nBetaRequested = 9;            % beta位点数量：固定为9个理论位点
fprintf('交叉验证折数设置为: %d\n', K_fold);
fprintf('beta位点数量设置为: %d\n', nBetaRequested);

% 只处理Python批量生成的模拟数据集，避免误读CTGAN_100datasets_scores.csv等汇总文件
files = dir(fullfile(dataFolder, 'synthetic_CTGAN_*.csv'));

summaryFile = fullfile(resultFolder, 'best_summary.csv');
if exist(summaryFile, 'file')
    delete(summaryFile);
end

fid = fopen(summaryFile, 'w');
fprintf(fid, 'filename,best_group,best_beta,best_eta,best_CV_F1,best_CV_AUC,best_CV_precision,best_CV_recall,best_CV_accuracy,best_CV_TP,best_CV_FP,best_CV_FN,best_CV_TN,best_CV_score,best_train_F1,best_train_AUC,best_train_precision,best_train_recall,best_train_accuracy,best_train_TP,best_train_FP,best_train_FN,best_train_TN,best_test_F1,best_test_AUC,best_test_precision,best_test_recall,best_test_accuracy,best_test_F1_stage1,best_test_AUC_stage1,best_test_precision_stage1,best_test_recall_stage1,best_test_accuracy_stage1,best_test_F1_stage2,best_test_AUC_stage2,best_test_precision_stage2,best_test_recall_stage2,best_test_accuracy_stage2\n');
fclose(fid);

%% ============================================
%   可视化风格设置（学术风格：白底、细网格、无衬线字体）
% ============================================
set(0, 'DefaultFigureColor', 'w');
set(0, 'DefaultAxesColor', 'w');
set(0, 'DefaultAxesGridColor', [0.88 0.88 0.88]);
set(0, 'DefaultAxesGridAlpha', 1);
set(0, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultAxesFontSize', FONT_TICK);
set(0, 'DefaultAxesXColor', [0.15 0.15 0.15]);
set(0, 'DefaultAxesYColor', [0.15 0.15 0.15]);
set(0, 'DefaultTextFontName', 'Times New Roman');
set(0, 'DefaultTextFontSize', FONT_TEXT);
set(0, 'DefaultLegendFontSize', FONT_LEGEND);
set(0, 'DefaultAxesLineWidth', 0.5);
set(0, 'DefaultAxesTickLength', [0.01 0.025]);

fprintf('发现 %d 个 CSV 文件，开始批量处理...\n', length(files));

% 用于统计所有文件的best_group对应的指标
all_best_group_vars = {};  % 存储每个文件best_group对应的指标名称
all_best_group_theta = {};  % 存储每个文件best_group对应的theta权重
all_best_group_var_indices = {};  % 存储每个文件best_group对应的指标索引
all_file_names = {};  % 存储所有文件名
all_var_names_full = {};  % 存储每个文件的完整varNames（用于获取指标名称）
% 用于存储所有CSV文件的best_group对应的测试集性能
all_best_test_F1 = [];          % 存储每个文件best_group对应的测试集F1
all_best_test_AUC = [];         % 存储每个文件best_group对应的测试集AUC
all_best_test_precision = [];   % 存储每个文件best_group对应的测试集precision
all_best_test_recall = [];      % 存储每个文件best_group对应的测试集recall
all_best_test_accuracy = [];    % 存储每个文件best_group对应的测试集accuracy
all_beta_test_F1_by_file = [];     % 行=synthetic_CTGAN文件，列=beta位点
all_beta_test_recall_by_file = []; % 行=synthetic_CTGAN文件，列=beta位点
all_best_test_n_samples = [];   % 存储每个文件测试集样本数（用于加权汇总）
all_best_test_PR_curves = {};  % 存储每个文件best_group对应的PR曲线数据
all_best_test_PR_AUC = [];      % 存储每个文件best_group对应的PR-AUC
all_best_test_ROC_curves = {};  % 存储每个文件best_group对应的ROC曲线数据
% 用于存储所有CSV文件的best_group对应的交叉验证性能
all_best_CV_AUC = [];  % 存储每个文件best_group对应的交叉验证平均AUC
all_best_CV_AUC_std = [];  % 存储每个文件best_group对应的交叉验证AUC标准差
all_best_CV_F1 = [];   % 存储每个文件best_group对应的交叉验证平均F1
all_best_CV_F1_std = [];   % 存储每个文件best_group对应的交叉验证F1标准差
% 用于收集所有CSV文件的个体级别数据（用于计算整体综合性能）
all_individual_scores = [];  % 存储所有样本的score
all_individual_labels = [];  % 存储所有样本的true_label
all_individual_cluster_ids = [];  % 存储所有样本的cluster_id（CSV文件编号）
all_individual_sample_ids = {};  % 存储所有样本的sample_id（文件名_样本索引）
all_individual_original_indices = [];  % 存储所有样本在原始数据中的索引（用于去重）
all_individual_file_names = {};  % 存储所有样本所属的文件名（用于去重）
all_individual_features = [];  % 存储所有样本的特征值（用于基于特征值去重）
% ===============================
% 汇总所有cluster混淆矩阵
% ===============================
all_TP = [];
all_FP = [];
all_FN = [];
all_TN = [];
all_individual_scores = [];
all_individual_labels = [];
%% ============================================
%   批量处理每个 CSV 文件
% ============================================
for fi = 1:length(files)
    fileName = files(fi).name;
    filePath = fullfile(dataFolder, fileName);
    file_TP = [];
    file_FP = [];
    file_FN = [];
    file_TN = [];
    file_individual_scores = [];
    file_individual_labels = [];
    % 训练集micro汇总：用于写入best_train_F1与best_train系列
    file_train_TP = [];
    file_train_FP = [];
    file_train_FN = [];
    file_train_TN = [];
    file_train_individual_scores = [];
    file_train_individual_labels = [];
    % 交叉验证micro汇总：每个cluster先合并K折验证集TP/FP/FN/TN，再在文件内合并cluster。
    file_cv_TP = [];
    file_cv_FP = [];
    file_cv_FN = [];
    file_cv_TN = [];
    file_cv_individual_scores = [];
    file_cv_individual_labels = [];

    % beta位点级文件内micro汇总：每个beta位点合并当前synthetic_CTGAN文件内所有有效cluster
    file_beta_TP = zeros(nBetaRequested, 1);
    file_beta_FP = zeros(nBetaRequested, 1);
    file_beta_FN = zeros(nBetaRequested, 1);
    file_beta_TN = zeros(nBetaRequested, 1);
    file_beta_CV_TP = zeros(nBetaRequested, 1);
    file_beta_CV_FP = zeros(nBetaRequested, 1);
    file_beta_CV_FN = zeros(nBetaRequested, 1);
    file_beta_CV_TN = zeros(nBetaRequested, 1);
    file_beta_train_TP = zeros(nBetaRequested, 1);
    file_beta_train_FP = zeros(nBetaRequested, 1);
    file_beta_train_FN = zeros(nBetaRequested, 1);
    file_beta_train_TN = zeros(nBetaRequested, 1);
    file_beta_scores = cell(nBetaRequested, 1);
    file_beta_labels = cell(nBetaRequested, 1);
    file_beta_CV_scores = cell(nBetaRequested, 1);
    file_beta_CV_labels = cell(nBetaRequested, 1);
    file_beta_train_scores = cell(nBetaRequested, 1);
    file_beta_train_labels = cell(nBetaRequested, 1);
    for beta_init_i = 1:nBetaRequested
        file_beta_scores{beta_init_i} = [];
        file_beta_labels{beta_init_i} = [];
        file_beta_CV_scores{beta_init_i} = [];
        file_beta_CV_labels{beta_init_i} = [];
        file_beta_train_scores{beta_init_i} = [];
        file_beta_train_labels{beta_init_i} = [];
    end

    fprintf('\n===============================\n');
    fprintf('正在处理文件：%s\n', fileName);
    fprintf('===============================\n');

    %% ==== 载入数据 ====
    zdata2024 = readtable(filePath, 'VariableNamingRule', 'preserve');

    %% ==== 数据划分：70%训练集，30%测试集 ====
    n_total = height(zdata2024);
    group_all = table2array(zdata2024(:,1));
    
    % 分层抽样：保持类别比例
    idx_class0 = find(group_all == 0);
    idx_class1 = find(group_all == 1);
    
    n_class0 = length(idx_class0);
    n_class1 = length(idx_class1);
    
    % 计算训练集和测试集大小（保持比例，训练:测试 = 70%:30%）
    n_train_class0 = round(n_class0 * 0.70);
    n_train_class1 = round(n_class1 * 0.70);
    % 确保测试集至少有1个样本（若总数很小）
    n_test_class0 = max(n_class0 - n_train_class0, 1);
    n_test_class1 = max(n_class1 - n_train_class1, 1);
    % 若需要，回调训练集数量以满足总数
    n_train_class0 = n_class0 - n_test_class0;
    n_train_class1 = n_class1 - n_test_class1;
    
    % 随机打乱索引（随机数种子已在文件开头设置）
    idx_class0_shuffled = idx_class0(randperm(n_class0));
    idx_class1_shuffled = idx_class1(randperm(n_class1));
    
    % 划分训练集和测试集
    train_idx_class0 = idx_class0_shuffled(1:n_train_class0);
    test_idx_class0 = idx_class0_shuffled(n_train_class0+1:end);
    
    train_idx_class1 = idx_class1_shuffled(1:n_train_class1);
    test_idx_class1 = idx_class1_shuffled(n_train_class1+1:end);
    
    train_idx = sort([train_idx_class0; train_idx_class1]);
    test_idx = sort([test_idx_class0; test_idx_class1]);
    
    % 提取训练集和测试集
    zdata2024_train = zdata2024(train_idx, :);
    zdata2024_test = zdata2024(test_idx, :);
    
    fprintf('数据划分完成：训练集 %d 个样本 (%.1f%%), 测试集 %d 个样本 (%.1f%%)\n', ...
        length(train_idx), length(train_idx)/n_total*100, ...
        length(test_idx), length(test_idx)/n_total*100);

    %% ==== GMM聚类：先划分训练/测试，再仅用训练集拟合GMM，并按概率密度归类测试集 ====
    originalFileName = fileName;
    originalBaseName = erase(originalFileName, '.csv');
    original_zdata2024_train = zdata2024_train;
    original_zdata2024_test = zdata2024_test;

    gmmFolder = fullfile(resultFolder, 'GMM_after_split');
    gmmResult = GMM_Clustering(original_zdata2024_train, original_zdata2024_test, ...
        'MinK', 3, ...
        'MaxK', 6, ...
        'MinTrainPositivePerCluster', 12, ...
        'RandomSeed', randomSeed, ...
        'BaseName', originalBaseName, ...
        'OutputDir', gmmFolder);

    for gmm_cluster_id = 1:gmmResult.bestK
        zdata2024_train = gmmResult.train_tables{gmm_cluster_id};
        zdata2024_test = gmmResult.test_tables{gmm_cluster_id};

        if height(zdata2024_train) < K_fold || height(zdata2024_test) < 1
            fprintf('跳过 %s cluster%d：训练集或测试集样本不足。\n', originalFileName, gmm_cluster_id);
            continue;
        end

        group_train_tmp = table2array(zdata2024_train(:, 1));
        n_train_class0_tmp = sum(group_train_tmp == 0);
        n_train_class1_tmp = sum(group_train_tmp == 1);
        min_class_count_tmp = min(n_train_class0_tmp, n_train_class1_tmp);

        if numel(unique(group_train_tmp)) < 2 || n_train_class1_tmp < 5 || min_class_count_tmp < 2
            fprintf('跳过 %s cluster%d：训练集中类别数不足。target0=%d, target1=%d，要求正类>=5且每类至少2个样本。\n', ...
                originalFileName, gmm_cluster_id, n_train_class0_tmp, n_train_class1_tmp);
            continue;
        end

        K_fold = min(K_fold_requested, min_class_count_tmp);
        if K_fold < K_fold_requested
            fprintf('提示：%s cluster%d 少数类训练样本数为%d，交叉验证折数由%d自适应调整为%d。\n', ...
                originalFileName, gmm_cluster_id, min_class_count_tmp, K_fold_requested, K_fold);
        end

        group_test_tmp = table2array(zdata2024_test(:, 1));
        if numel(unique(group_test_tmp)) < 2
            fprintf('提示：%s cluster%d 测试集仅包含单一类别，部分测试指标可能为NaN或退化。\n', originalFileName, gmm_cluster_id);
        end

        fileName = sprintf('%s_cluster%d.csv', originalBaseName, gmm_cluster_id);
        train_idx = (1:height(zdata2024_train))';
        test_idx = (1:height(zdata2024_test))';
        train_idx_class0 = find(table2array(zdata2024_train(:, 1)) == 0);
        train_idx_class1 = find(table2array(zdata2024_train(:, 1)) == 1);
        test_idx_class0 = find(table2array(zdata2024_test(:, 1)) == 0);
        test_idx_class1 = find(table2array(zdata2024_test(:, 1)) == 1);
        n_total = height(zdata2024_train) + height(zdata2024_test);

        fprintf('\n开始 DA-DEA：%s | train=%d, test=%d\n', ...
            fileName, height(zdata2024_train), height(zdata2024_test));

    %% ==== 特征分组（仅基于训练集数据） ====
    clear opts
    opts = optimoptions('linprog','Algorithm','dual-simplex','Display','none');

    % 仅使用训练集数据进行特征分组
    Z_train = table2array(zdata2024_train(:,2:end));
    varNames = zdata2024_train.Properties.VariableNames(2:end);
    % 保存原始变量名，确保可视化时使用
    original_varNames = varNames;
    [R, ~] = corrcoef(Z_train, 'Rows', 'pairwise');
    threshold = 0.9;

    numVars = size(Z_train, 2);
    groups = {};
    assigned = false(1, numVars);

    for i = 1:numVars
        newGroup = i;
        assigned(i) = true;
        for j = 1:numVars
            corrWithGroup = abs(R(j, newGroup));
            if all(corrWithGroup <= threshold)
                newGroup = [newGroup, j];
                assigned(j) = true;
            end
        end
        groups{end+1} = newGroup;
    end

    for i = 1:numel(groups)
        if ~isempty(groups{i})
            groups{i} = sort(groups{i}, 'ascend');
        end
    end

    groupStrings = cellfun(@(x) num2str(x),  groups, 'UniformOutput', false);
    [~, uniqueIdx] = unique(groupStrings, 'stable');
    groups_unique = groups(uniqueIdx);
    nGroups = numel(groups_unique);

    fprintf('特征分组完成：共 %d 个特征组（仅基于训练集）\n', nGroups);

    %% ==== 基于特征分组提取训练集的特征组 ====
    group = table2array(zdata2024_train(:,1));
    Z = table2array(zdata2024_train(:,2:end));
    
    for i = 1:nGroups
        colIdx = groups_unique{i}; 
        varName = sprintf('Z%d', i);
        Zdata = table2array(zdata2024_train(:, colIdx+1));
        assignin('base', varName, Zdata);
    end

    %% ==== 创建指标正负性映射表 ====
    % +1: 良性指标（越高越好），-1: 恶性P_LCR指标（越高越危险），0: 不定项
    indicator_direction = containers.Map('KeyType', 'char', 'ValueType', 'double');
    
    % 设置指标正负性
% 负向指标
indicator_direction('Neu_pct')  = -1;
indicator_direction('Neu_abs')  = -1;
indicator_direction('RDW_CV')   = -1;
indicator_direction('RDW_SD')   = -1;
indicator_direction('FIB')      = -1;
indicator_direction('D_Dimer')  = -1;
indicator_direction('PDW')      = -1;
indicator_direction('PTR')      = -1;
indicator_direction('INR')      = -1;
indicator_direction('PT')       = -1;
indicator_direction('TT')       = -1;

% 正向指标
indicator_direction('RBC')      = 1;
indicator_direction('Hb')       = 1;
indicator_direction('HCT')      = 1;
indicator_direction('MCHC')     = 1;
indicator_direction('PLT')      = 0;
indicator_direction('PT_pct')   = 1;

% 不定项指标
indicator_direction('WBC')      = -1;
indicator_direction('Age')      = 0;
indicator_direction('Gen')      = 0;
indicator_direction('Eos_pct')  = 0;
indicator_direction('Bas_abs')  = 0;
indicator_direction('Bas_pct')  = 0;
indicator_direction('Eos_abs')  = 0;
indicator_direction('PCT')      = 0;
indicator_direction('APTT')     = 0;
indicator_direction('Mon_pct')  = 0;
indicator_direction('Mon_abs')  = 0;
indicator_direction('MPV')      = 0;
indicator_direction('MCV')      = 0;
indicator_direction('Lym_pct')  = 0;
indicator_direction('MCH')      = 0;
indicator_direction('P_LCR')    = 0;

    fixed_eta = 1;
    counts = histcounts(group, [-1 0.5 2]);

    %% ========== beta 9个位点遍历：1 + beta0 + beta0到g1/g2的几何展开 ==========
    % 设计思路：
    %   (1) beta = 1：保留已有文献常用的固定权重；
    %   (2) beta0：本文根据类别不平衡程度定义的自适应权重；
    %   (3) 从 beta0 到 g1/g2 采用等比（几何）尺度展开，最终覆盖 beta = g1/g2。
    %
    % 定义：
    %   r = g1/g2 = n0/n1
    %   x = max(0, log10(r/9))
    %   beta0 = 1 + x/(1+x)
    %   q = (r/beta0)^(1/7)
    %
    % 9个位点：
    %   [1, beta0, beta0*q, beta0*q^2, ..., beta0*q^7]
    % 其中 beta0*q^7 = r = g1/g2。

    r = counts(1,1) / max(counts(1,2), eps);
    r = max(r, 1);   % 当前研究中g1为多数类；保护性限制，避免出现r<1导致反向展开

    x = max(0, log10(r / 9));
    beta0 = 1 + x / (1 + x);

    q = (r / beta0)^(1/7);

    betaGrid = [ ...
        1, ...
        beta0, ...
        beta0 * q, ...
        beta0 * q^2, ...
        beta0 * q^3, ...
        beta0 * q^4, ...
        beta0 * q^5, ...
        beta0 * q^6, ...
        beta0 * q^7];

    % 消除最后一个点的浮点累计误差，严格令第9点等于g1/g2
    betaGrid(end) = r;
    nBeta = numel(betaGrid);

    % 横坐标标签：突出三个有直接理论含义的位置
    % 第1点：beta=1；第2点：本文beta0；第9点：beta=g1/g2
    beta_tick_labels = {'1', '\beta_0', '', '', '', '', '', '', 'g_1/g_2'};

    fprintf('类别不平衡比 r=g1/g2 = %.6f\n', r);
    fprintf('自适应 beta0 = %.6f\n', beta0);
    fprintf('几何展开公比 q = %.6f\n', q);
    fprintf('beta 9个位点：');
    fprintf(' %.6f', betaGrid);
    fprintf('\n beta位点数量=%d\n', nBeta);

    K_fold_current = min(K_fold_requested, min(counts));
    if K_fold_current < 2
        fprintf('跳过 %s：当前cluster训练集中少数类样本数为%d，无法进行分层交叉验证。\n', fileName, min(counts));
        continue;
    end
    K_fold = K_fold_current;

    %% =====================================================
    %   生成K折交叉验证索引（分层抽样）
    % =====================================================
    n_train = length(group);
    idx_class0_train = find(group == 0);
    idx_class1_train = find(group == 1);
    n_class0_train = length(idx_class0_train);
    n_class1_train = length(idx_class1_train);
    
    % 为每个类别生成K折索引
    cv_idx_class0 = cell(K_fold, 1);
    cv_idx_class1 = cell(K_fold, 1);
    
    % 打乱索引
    idx_class0_shuffled = idx_class0_train(randperm(n_class0_train));
    idx_class1_shuffled = idx_class1_train(randperm(n_class1_train));
    
    % 计算每折的样本数
    fold_size_class0 = floor(n_class0_train / K_fold);
    fold_size_class1 = floor(n_class1_train / K_fold);
    
    % 分配每折的索引
    for k = 1:K_fold
        if k < K_fold
            start_idx0 = (k-1) * fold_size_class0 + 1;
            end_idx0 = k * fold_size_class0;
            start_idx1 = (k-1) * fold_size_class1 + 1;
            end_idx1 = k * fold_size_class1;
        else
            % 最后一折包含剩余所有样本
            start_idx0 = (k-1) * fold_size_class0 + 1;
            end_idx0 = n_class0_train;
            start_idx1 = (k-1) * fold_size_class1 + 1;
            end_idx1 = n_class1_train;
        end
        cv_idx_class0{k} = idx_class0_shuffled(start_idx0:end_idx0);
        cv_idx_class1{k} = idx_class1_shuffled(start_idx1:end_idx1);
    end
    
    fprintf('K折交叉验证索引生成完成（K=%d）\n', K_fold);

    %% =====================================================
    %   对每个Z_group进行K折交叉验证
    % =====================================================
    results = struct();
    idx = 1;

    total_iter = nGroups * nBeta * K_fold;
    h = waitbar(0, sprintf('运行文件 %s (K折交叉验证，beta固定)...', fileName));

    for g = 1:nGroups
        Z_group = eval(sprintf('Z%d', g));

        % 根据当前分组包含的指标设置xi_input正负性限制
        xi_input = zeros(1, size(Z_group, 2));
        if g <= numel(groups_unique)
            group_var_indices = groups_unique{g};
            for v = 1:length(group_var_indices)
                if v <= length(xi_input)
                    var_idx = group_var_indices(v);
                    if var_idx <= length(varNames)
                        var_name = varNames(var_idx);
                        % 转换为字符串
                        if iscell(var_name)
                            var_name_str = char(var_name{1});
                        elseif isstring(var_name)
                            var_name_str = char(var_name);
                        else
                            var_name_str = char(string(var_name));
                        end
                        
                        % 查找匹配的指标方向（仅精确匹配）
                        var_name_clean = strtrim(var_name_str);  % 去除首尾空格
                        if isKey(indicator_direction, var_name_clean)
                            xi_input(v) = indicator_direction(var_name_clean);
                        end
                    end
                end
            end
        end
        
        xi_p = (xi_input == 1) | (xi_input == 0);
        xi_m = (xi_input == -1) | (xi_input == 0);

        for beta_i = 1:nBeta
            beta = betaGrid(beta_i);
            eta = fixed_eta;
            
            % K折交叉验证
            cv_F1 = zeros(K_fold, 1);
            cv_AUC = zeros(K_fold, 1);
            cv_Accuracy = zeros(K_fold, 1);
            cv_precision = zeros(K_fold, 1);
            cv_recall = zeros(K_fold, 1);
            cv_sol = cell(K_fold, 1);
            cv_TP = zeros(K_fold, 1);
            cv_FP = zeros(K_fold, 1);
            cv_FN = zeros(K_fold, 1);
            cv_TN = zeros(K_fold, 1);
            cv_scores_all = [];
            cv_labels_all = [];
            
            for k = 1:K_fold
                % 生成验证集索引（当前折）
                val_idx = sort([cv_idx_class0{k}; cv_idx_class1{k}]);
                
                % 生成训练集索引（其他所有折）
                train_cv_idx = [];
                for kk = 1:K_fold
                    if kk ~= k
                        train_cv_idx = [train_cv_idx; cv_idx_class0{kk}; cv_idx_class1{kk}];
                    end
                end
                train_cv_idx = sort(train_cv_idx);
                
                % 提取交叉验证的训练集和验证集
                Z_group_train_cv = Z_group(train_cv_idx, :);
                group_train_cv = group(train_cv_idx);
                Z_group_val_cv = Z_group(val_idx, :);
                group_val_cv = group(val_idx);
                
                % ============================================================
                %  第一阶段：在训练折上学习 θ, d, c
                % ============================================================
                [R1_train_cv, R2_train_cv, R0_train_cv, sol_train_cv] = model8_1_7(Z_group_train_cv, group_train_cv, ...
                    beta, eta, xi_p, xi_m, opts);
                % ===== 自适应 eta =====
%                 scores_train_cv = sol_train_cv.theta.' * Z_group_train_cv.';
%                 sigma_cv = std(scores_train_cv);
%                 
%                 kappa = 0.35;   % 推荐 0.3~0.4
%                 eta = kappa * sigma_cv;
                % 提取训练折的参数
                theta_cv = sol_train_cv.theta;
                d_cv = sol_train_cv.d;
                
                % 从第二阶段模型中提取 c 和 theta（如果存在）
                if isfield(sol_train_cv, 'submodel') && isfield(sol_train_cv.submodel, 'c')
                    c_cv = sol_train_cv.submodel.c;
                    theta_stage2_cv = sol_train_cv.submodel.theta;  % 第二阶段优化后的 theta
                else
                    c_cv = d_cv - eta / 2;
                    theta_stage2_cv = theta_cv;  % 无第二阶段时使用第一阶段 theta
                end
                
                % ============================================================
                %  第二阶段：固定 θ, d, c，在验证折上预测
                % ============================================================
                % 计算验证集 scores = θᵀx（第一阶段）
                Z_group_val_cv_T = Z_group_val_cv.';
                scores_val = theta_cv.' * Z_group_val_cv_T;
                scores_val = scores_val(:);
                
                % 按规则分类
                R1_val = find(scores_val > d_cv);
                R2_val = find(scores_val < d_cv - eta);
                
                % R0 区域：d - eta <= score <= d
                R0_candidates = find((scores_val >= d_cv - eta) & (scores_val <= d_cv));
                
                % 对 R0 区域进一步分类
                C1_val = [];
                C2_val = [];
                if ~isempty(R0_candidates)
                    % 使用第二阶段优化后的 theta 重新计算 R0 样本的 score
                    Z_R0 = Z_group_val_cv(R0_candidates, :);
                    scores_R0_stage2 = theta_stage2_cv.' * Z_R0.';  % 使用第二阶段 theta
                    scores_R0_stage2 = scores_R0_stage2(:);
                    
                    % 根据第二阶段 theta 计算的 score 划分 C1/C2
                    C1_val = R0_candidates(scores_R0_stage2 > c_cv);
                    C2_val = R0_candidates(scores_R0_stage2 <= c_cv);
                    R1_val = unique([R1_val; C1_val]);
                    R2_val = unique([R2_val; C2_val]);
                end
                R0_val = setdiff(1:length(group_val_cv), union(R1_val, R2_val));
                
                % ============================================================
                %  计算验证集性能指标
                % ============================================================
                y_true_val = group_val_cv(:);
                y_pred_val = zeros(length(group_val_cv), 1);
                y_pred_val(R1_val) = 0;
                y_pred_val(R2_val) = 1;
                
                y_true_bin = (y_true_val == 1);
                y_pred_bin = (y_pred_val == 1);
                
                TP = sum(y_true_bin == 1 & y_pred_bin == 1);
                FP = sum(y_true_bin == 0 & y_pred_bin == 1);
                FN = sum(y_true_bin == 1 & y_pred_bin == 0);
                TN = sum(y_true_bin == 0 & y_pred_bin == 0);
                cv_TP(k) = TP;
                cv_FP(k) = FP;
                cv_FN(k) = FN;
                cv_TN(k) = TN;
                
                precision_val = TP / (TP + FP + eps);
                recall_val = TP / (TP + FN + eps);
                F1_val = 2 * precision_val * recall_val / (precision_val + recall_val + eps);
                Accuracy_val = (TP + TN) / (TP + TN + FP + FN + eps);
                
                % AUC 计算
                scores_auc = -scores_val;
                cv_scores_all = [cv_scores_all; scores_auc(:)];
                cv_labels_all = [cv_labels_all; double(y_true_bin(:))];
                ranks = tiedrank(scores_auc);
                pos_idx = find(y_true_bin == 1);
                neg_idx = find(y_true_bin == 0);
                m = length(pos_idx);
                n_pos = length(neg_idx);
                if m > 0 && n_pos > 0
                    S = sum(ranks(pos_idx));
                    AUC_val = (S - m*(m+1)/2) / (m * n_pos + eps);
                else
                    AUC_val = NaN;
                end
                
                % 保存当前折的性能指标
                cv_F1(k) = F1_val;
                cv_AUC(k) = AUC_val;
                cv_Accuracy(k) = Accuracy_val;
                cv_precision(k) = precision_val;
                cv_recall(k) = recall_val;
                cv_sol{k} = sol_train_cv;
                
                % 更新进度条
                current_iter = ((g-1) * nBeta + (beta_i-1)) * K_fold + k;
                waitbar(current_iter / total_iter, h, ...
                    sprintf('文件 %s 进度 %.1f%% (Group %d/%d, Fold %d/%d, beta=%.6f)', ...
                    fileName, current_iter / total_iter * 100, g, nGroups, k, K_fold, beta));
            end
            
            % 计算K折交叉验证的micro性能：先合并各验证折TP/FP/FN/TN，再计算指标。
            cv_TP_total = sum(cv_TP);
            cv_FP_total = sum(cv_FP);
            cv_FN_total = sum(cv_FN);
            cv_TN_total = sum(cv_TN);
            cv_precision_micro = cv_TP_total / (cv_TP_total + cv_FP_total + eps);
            cv_recall_micro = cv_TP_total / (cv_TP_total + cv_FN_total + eps);
            cv_F1_micro = 2 * cv_precision_micro * cv_recall_micro / (cv_precision_micro + cv_recall_micro + eps);
            cv_Accuracy_micro = (cv_TP_total + cv_TN_total) / (cv_TP_total + cv_FP_total + cv_FN_total + cv_TN_total + eps);
            if ~isempty(cv_scores_all)
                cv_pos_all = cv_labels_all == 1;
                cv_neg_all = cv_labels_all == 0;
                if sum(cv_pos_all) > 0 && sum(cv_neg_all) > 0
                    cv_ranks_all = tiedrank(cv_scores_all);
                    cv_m_all = sum(cv_pos_all);
                    cv_n_all = sum(cv_neg_all);
                    cv_S_all = sum(cv_ranks_all(cv_pos_all));
                    cv_AUC_micro = (cv_S_all - cv_m_all*(cv_m_all+1)/2) / (cv_m_all * cv_n_all + eps);
                else
                    cv_AUC_micro = NaN;
                end
            else
                cv_AUC_micro = NaN;
            end

            % 计算K折交叉验证性能
            results(idx).group = g;
            results(idx).eta = eta;
            results(idx).beta = beta;
            results(idx).F1 = cv_F1_micro;
            results(idx).AUC = cv_AUC_micro;
            results(idx).Accuracy = cv_Accuracy_micro;
            results(idx).precision = cv_precision_micro;
            results(idx).recall = cv_recall_micro;
            results(idx).F1_std = std(cv_F1);
            results(idx).AUC_std = std(cv_AUC);
            results(idx).Accuracy_std = std(cv_Accuracy);
            results(idx).precision_std = std(cv_precision);
            results(idx).recall_std = std(cv_recall);
            results(idx).TP = cv_TP_total;
            results(idx).FP = cv_FP_total;
            results(idx).FN = cv_FN_total;
            results(idx).TN = cv_TN_total;
            results(idx).scores = cv_scores_all;
            results(idx).labels = cv_labels_all;
            % 保存最后一折的sol作为代表（用于后续可视化）
            results(idx).sol = cv_sol{K_fold};
            results(idx).theta = cv_sol{K_fold}.theta;
            idx = idx + 1;
        end
    end

    close(h);

    %% =====================================================
    %   提取结果向量（F1 / AUC / recall / precision）
    % =====================================================
    F1_values = zeros(nGroups, 1);
    AUC_values = zeros(nGroups, 1);
    recall_values = zeros(nGroups, 1);
    precision_values = zeros(nGroups, 1);

    for g = 1:nGroups
        idx_group = find([results.group] == g);
        if ~isempty(idx_group)
            [~, idx_group_best_local] = max([results(idx_group).F1]);
            idx_group_best = idx_group(idx_group_best_local);
            F1_values(g) = results(idx_group_best).F1;
            AUC_values(g) = results(idx_group_best).AUC;
            recall_values(g) = results(idx_group_best).recall;
            precision_values(g) = results(idx_group_best).precision;
        end
    end


    baseName = erase(fileName, '.csv');

    %% =====================================================
    %   计算全局最优参数（基于交叉验证平均性能）
    % =====================================================
    F1_all = [results.F1];
    AUC_all = [results.AUC];
    Accuracy_all = [results.Accuracy];
    precision_all = [results.precision];
    recall_all = [results.recall];
    F1_std_all = [results.F1_std];
    AUC_std_all = [results.AUC_std];
    Accuracy_std_all = [results.Accuracy_std];
    precision_std_all = [results.precision_std];
    recall_std_all = [results.recall_std];
    eta_all = [results.eta];
    beta_all = [results.beta];
    group_all = [results.group];

    % 综合评分：平衡区分能力、正类识别能力、整体稳定性
    % 主指标采用加权调和平均，避免某一项很高但其他指标明显偏低的模型被选中；
    % 同时用F1/AUC的交叉验证标准差作为稳定性惩罚。
    metric_matrix = [F1_all(:), AUC_all(:), recall_all(:), precision_all(:), Accuracy_all(:)];
    metric_matrix(~isfinite(metric_matrix)) = 0;
    metric_matrix = max(min(metric_matrix, 1), 0);

    score_weights = [0.30, 0.25, 0.25, 0.10, 0.10];  % F1, AUC, Recall, Precision, Accuracy
    harmonic_score = sum(score_weights) ./ sum(score_weights ./ max(metric_matrix, eps), 2);

    stability_penalty = 0.10 * F1_std_all(:) + 0.05 * AUC_std_all(:);
    Score_all = harmonic_score - stability_penalty;
    Score_all(~isfinite(Score_all)) = -Inf;

    [best_score, idxMax] = max(Score_all);

    best_F1        = F1_all(idxMax);
    best_AUC       = AUC_all(idxMax);
    best_precision = precision_all(idxMax);
    best_recall    = recall_all(idxMax);
    best_accuracy  = Accuracy_all(idxMax);
    best_F1_std    = F1_std_all(idxMax);
    best_AUC_std   = AUC_std_all(idxMax);
    best_accuracy_std = Accuracy_std_all(idxMax);
    best_precision_std = precision_std_all(idxMax);
    best_recall_std = recall_std_all(idxMax);
    best_beta      = beta_all(idxMax);
    best_eta       = eta_all(idxMax);
    best_group     = group_all(idxMax);
    best_CV_TP     = results(idxMax).TP;
    best_CV_FP     = results(idxMax).FP;
    best_CV_FN     = results(idxMax).FN;
    best_CV_TN     = results(idxMax).TN;

    file_cv_TP(end+1) = best_CV_TP;
    file_cv_FP(end+1) = best_CV_FP;
    file_cv_FN(end+1) = best_CV_FN;
    file_cv_TN(end+1) = best_CV_TN;
    file_cv_individual_scores = [file_cv_individual_scores; results(idxMax).scores(:)];
    file_cv_individual_labels = [file_cv_individual_labels; results(idxMax).labels(:)];
    
    fprintf('交叉验证最佳参数：group=%d, beta=%.6f, eta=%.6f\n', ...
        best_group, best_beta, best_eta);
    fprintf('交叉验证micro性能：F1=%.4f±%.4f, AUC=%.4f±%.4f, Precision=%.4f±%.4f, Recall=%.4f±%.4f, Accuracy=%.4f±%.4f, 综合评分=%.4f\n', ...
        best_F1, best_F1_std, best_AUC, best_AUC_std, best_precision, best_precision_std, ...
        best_recall, best_recall_std, best_accuracy, best_accuracy_std, best_score);

    %% =====================================================
    %   准备测试集数据（用于best_group评估）
    % =====================================================
    group_test = table2array(zdata2024_test(:,1));
    Z_test = table2array(zdata2024_test(:,2:end));

    %% =====================================================
    %   beta位点级结果：每个beta位点独立选择best_group，并在完整训练集重训后测试
    %   输出口径：
    %   1) cluster级：synthetic_CTGAN_xxx_clusterx_metrics.csv，除Metric列外，每列为Beta_01...Beta_09
    %   2) 文件级：synthetic_CTGAN_xxx_global_micro_metrics.csv，同样按Beta_01...Beta_09展开
    % =====================================================
    betaMetricNames = {'best_CV_F1','best_CV_AUC','best_CV_precision','best_CV_recall','best_CV_accuracy', ...
                       'best_CV_TP','best_CV_FP','best_CV_FN','best_CV_TN', ...
                       'best_train_F1','best_train_AUC','best_train_precision','best_train_recall','best_train_accuracy', ...
                       'best_train_TP','best_train_FP','best_train_FN','best_train_TN', ...
                       'best_test_F1','best_test_AUC','best_test_precision','best_test_recall','best_test_accuracy', ...
                       'best_test_TP','best_test_FP','best_test_FN','best_test_TN', ...
                       'best_group','best_eta','best_beta','best_CV_score'};
    clusterBetaValues = NaN(numel(betaMetricNames), nBeta);

    for beta_pos = 1:nBeta
        beta_value = betaGrid(beta_pos);
        idx_beta_candidates = find(abs(beta_all - beta_value) <= 1e-12 | (isnan(beta_all) & isnan(beta_value)));

        if isempty(idx_beta_candidates)
            continue;
        end

        [~, beta_local_best_pos] = max(Score_all(idx_beta_candidates));
        idx_beta_best = idx_beta_candidates(beta_local_best_pos);

        beta_group = group_all(idx_beta_best);
        beta_eta = eta_all(idx_beta_best);
        beta_best_score = Score_all(idx_beta_best);

        if beta_group > numel(groups_unique)
            continue;
        end

        beta_group_cols = groups_unique{beta_group};
        Z_train_beta_best = Z(:, beta_group_cols);
        Z_test_beta_best = Z_test(:, beta_group_cols);

        try
            beta_eval = evaluate_dadea_final_model(Z_train_beta_best, group, Z_test_beta_best, group_test, ...
                beta_value, beta_eta, beta_group_cols, varNames, indicator_direction, opts);

            clusterBetaValues(:, beta_pos) = [ ...
                results(idx_beta_best).F1; results(idx_beta_best).AUC; results(idx_beta_best).precision; results(idx_beta_best).recall; results(idx_beta_best).Accuracy; ...
                results(idx_beta_best).TP; results(idx_beta_best).FP; results(idx_beta_best).FN; results(idx_beta_best).TN; ...
                beta_eval.train.F1; beta_eval.train.AUC; beta_eval.train.precision; beta_eval.train.recall; beta_eval.train.accuracy; ...
                beta_eval.train.TP; beta_eval.train.FP; beta_eval.train.FN; beta_eval.train.TN; ...
                beta_eval.test.F1; beta_eval.test.AUC; beta_eval.test.precision; beta_eval.test.recall; beta_eval.test.accuracy; ...
                beta_eval.test.TP; beta_eval.test.FP; beta_eval.test.FN; beta_eval.test.TN; ...
                beta_group; beta_eta; beta_value; beta_best_score];

            file_beta_CV_TP(beta_pos) = file_beta_CV_TP(beta_pos) + results(idx_beta_best).TP;
            file_beta_CV_FP(beta_pos) = file_beta_CV_FP(beta_pos) + results(idx_beta_best).FP;
            file_beta_CV_FN(beta_pos) = file_beta_CV_FN(beta_pos) + results(idx_beta_best).FN;
            file_beta_CV_TN(beta_pos) = file_beta_CV_TN(beta_pos) + results(idx_beta_best).TN;
            file_beta_CV_scores{beta_pos} = [file_beta_CV_scores{beta_pos}; results(idx_beta_best).scores(:)];
            file_beta_CV_labels{beta_pos} = [file_beta_CV_labels{beta_pos}; results(idx_beta_best).labels(:)];

            file_beta_train_TP(beta_pos) = file_beta_train_TP(beta_pos) + beta_eval.train.TP;
            file_beta_train_FP(beta_pos) = file_beta_train_FP(beta_pos) + beta_eval.train.FP;
            file_beta_train_FN(beta_pos) = file_beta_train_FN(beta_pos) + beta_eval.train.FN;
            file_beta_train_TN(beta_pos) = file_beta_train_TN(beta_pos) + beta_eval.train.TN;
            file_beta_train_scores{beta_pos} = [file_beta_train_scores{beta_pos}; beta_eval.train.scores(:)];
            file_beta_train_labels{beta_pos} = [file_beta_train_labels{beta_pos}; beta_eval.train.labels(:)];

            file_beta_TP(beta_pos) = file_beta_TP(beta_pos) + beta_eval.test.TP;
            file_beta_FP(beta_pos) = file_beta_FP(beta_pos) + beta_eval.test.FP;
            file_beta_FN(beta_pos) = file_beta_FN(beta_pos) + beta_eval.test.FN;
            file_beta_TN(beta_pos) = file_beta_TN(beta_pos) + beta_eval.test.TN;
            file_beta_scores{beta_pos} = [file_beta_scores{beta_pos}; beta_eval.test.scores(:)];
            file_beta_labels{beta_pos} = [file_beta_labels{beta_pos}; beta_eval.test.labels(:)];
        catch ME_beta
            fprintf('  警告：%s beta位点%d最终评估失败：%s\n', fileName, beta_pos, ME_beta.message);
        end
    end

    cluster_metrics_file_beta = fullfile(resultFolder, [baseName, '_metrics.csv']);
    write_beta_metric_table(cluster_metrics_file_beta, betaMetricNames, clusterBetaValues, nBeta);

    %% =====================================================
    %   使用最佳参数在整个训练集上重新训练，然后在测试集上评估
    % =====================================================
    fprintf('\n使用最佳参数在整个训练集上重新训练，然后在测试集上评估：group=%d, beta=%.6f, eta=%.6f\n', ...
        best_group, best_beta, best_eta);
    
    % 获取最佳分组对应的Z列
    if best_group <= numel(groups_unique)
        best_group_cols = groups_unique{best_group};
        Z_group_best = eval(sprintf('Z%d', best_group));
    else
        error('best_group超出范围');
    end
    
    % 准备xi参数（与训练时相同，根据指标正负性设置）
    xi_input = zeros(1, size(Z_group_best, 2));
    if best_group <= numel(groups_unique)
        best_group_var_indices = groups_unique{best_group};
        for v = 1:length(best_group_var_indices)
            if v <= length(xi_input)
                var_idx = best_group_var_indices(v);
                if var_idx <= length(varNames)
                    var_name = varNames(var_idx);
                    % 转换为字符串
                    if iscell(var_name)
                        var_name_str = char(var_name{1});
                    elseif isstring(var_name)
                        var_name_str = char(var_name);
                    else
                        var_name_str = char(string(var_name));
                    end
                    
                    % 查找匹配的指标方向（仅精确匹配）
                    var_name_clean = strtrim(var_name_str);  % 去除首尾空格
                    if isKey(indicator_direction, var_name_clean)
                        xi_input(v) = indicator_direction(var_name_clean);
                    end
                end
            end
        end
    end
    
    xi_p = (xi_input == 1) | (xi_input == 0);
    xi_m = (xi_input == -1) | (xi_input == 0);
    
    % ============================================================
    %  第一阶段：在训练集上学习 θ, d, c
    % ============================================================
    % 在整个训练集上重新训练最佳模型
    [R1_train_final, R2_train_final, R0_train_final, sol_train_final] = model8_1_7(Z_group_best, group, ...
        best_beta, best_eta, xi_p, xi_m, opts);
    
    fprintf('训练集最终性能：F1=%.4f, AUC=%.4f, Precision=%.4f, Recall=%.4f, Accuracy=%.4f\n', ...
        sol_train_final.F1, sol_train_final.AUC, sol_train_final.precision, ...
        sol_train_final.recall, sol_train_final.Accuracy);
    
    % ============================================================
    %  提取训练好的参数：θ, d, c
    % ============================================================
    % 从训练好的模型中提取 theta
    theta_train = sol_train_final.theta;
    d_train = sol_train_final.d;
    
    % ===== 自适应 eta =====
    % 使用训练集的 theta 计算 scores，然后用标准差计算 eta
    scores_train_full = theta_train.' * Z_group_best.';
%     sigma_full = std(scores_train_full);
% 
%     kappa = 0.35;  % 推荐 0.3~0.4
%     eta_train = kappa * sigma_full;
    eta_train = best_eta;
    % 打印训练好的参数
%     fprintf('  训练集参数: d = %.4f, eta = %.4f (kappa=%.2f, sigma=%.4f)\n', d_train, eta_train, kappa, sigma_full);
    
    % 从第二阶段模型中提取 c（如果存在）
    if isfield(sol_train_final, 'submodel') && isfield(sol_train_final.submodel, 'c')
        c_train = sol_train_final.submodel.c;  % 全局标量 c
        fprintf('  第二阶段参数: c = %.4f (c0 = %.4f)\n', c_train, d_train - eta_train/2);
    else
        % 如果没有第二阶段，使用默认 c = d - eta/2
        c_train = d_train - eta_train / 2;
        fprintf('  无重叠区域，使用默认 c0 = %.4f\n', c_train);
    end

    % ============================================================
    %  计算完整训练集性能指标：用于文件级best_train系列汇总
    %  口径：使用model8_1_7内部返回的训练集划分与训练性能，不再外部二次划分R0/C1/C2。
    % ============================================================
    y_true_train = group(:);
    y_pred_train = zeros(length(group), 1);
    y_pred_train(R1_train_final) = 0;
    y_pred_train(R2_train_final) = 1;

    y_true_train_bin = (y_true_train == 1);
    y_pred_train_bin = (y_pred_train == 1);

    TP_train = sum(y_true_train_bin == 1 & y_pred_train_bin == 1);
    FP_train = sum(y_true_train_bin == 0 & y_pred_train_bin == 1);
    FN_train = sum(y_true_train_bin == 1 & y_pred_train_bin == 0);
    TN_train = sum(y_true_train_bin == 0 & y_pred_train_bin == 0);

    best_train_F1 = sol_train_final.F1;
    best_train_AUC = sol_train_final.AUC;
    best_train_precision = sol_train_final.precision;
    best_train_recall = sol_train_final.recall;
    best_train_accuracy = sol_train_final.Accuracy;

    train_scores_for_auc = -scores_train_full(:);

    file_train_TP(end+1) = TP_train;
    file_train_FP(end+1) = FP_train;
    file_train_FN(end+1) = FN_train;
    file_train_TN(end+1) = TN_train;
    file_train_individual_scores = [file_train_individual_scores; train_scores_for_auc(:)];
    file_train_individual_labels = [file_train_individual_labels; double(y_true_train_bin(:))];

    fprintf('训练集内部性能用于best_train：F1=%.4f, AUC=%.4f, Precision=%.4f, Recall=%.4f, Accuracy=%.4f\n', ...
        best_train_F1, best_train_AUC, best_train_precision, best_train_recall, best_train_accuracy);
    
    % ============================================================
    %  第二阶段：固定 θ, d, c，直接在测试集上分类
    % ============================================================
    % 获取最佳分组对应的Z_test列
    Z_test_group = Z_test(:, best_group_cols);
    
    % 计算测试集 scores = θᵀx
    Z_test_group_T = Z_test_group.';
    test_scores = theta_train.' * Z_test_group_T;
    test_scores = test_scores(:);
    
    % 按规则分类：
    %   score > d          → R1
    %   score < d - eta    → R2
    %   d - eta < score < d → R0，进一步用 c 分类
    %       score > c      → C1 (归属 R1)
    %       score < c      → C2 (归属 R2)
    
    R1_test = find(test_scores > d_train);
    R2_test = find(test_scores < d_train - eta_train);
    
    % R0 区域：d - eta <= score <= d
    R0_candidates = find((test_scores >= d_train - eta_train) & (test_scores <= d_train));
    
    % 对 R0 区域进一步分类
    C1_test = [];  % R0 中归属 R1 的样本
    C2_test = [];  % R0 中归属 R2 的样本
    theta_stage2 = theta_train;  % 默认使用第一阶段 theta
    if isfield(sol_train_final, 'submodel') && isfield(sol_train_final.submodel, 'theta')
        theta_stage2 = sol_train_final.submodel.theta;  % 使用第二阶段优化后的 theta
    end
    
    if ~isempty(R0_candidates)
        % 使用第二阶段优化后的 theta 重新计算 R0 样本的 score
        Z_R0_test = Z_test_group(R0_candidates, :);
        scores_R0_stage2 = theta_stage2.' * Z_R0_test.';  % 使用第二阶段 theta
        scores_R0_stage2 = scores_R0_stage2(:);
        
        % 根据第二阶段 theta 计算的 score 划分 C1/C2
        C1_test = R0_candidates(scores_R0_stage2 > c_train);  % score > c → C1 (R1)
        C2_test = R0_candidates(scores_R0_stage2 <= c_train); % score <= c → C2 (R2)
        
        % 将 C1 并入 R1，C2 并入 R2
        R1_test = unique([R1_test; C1_test]);
        R2_test = unique([R2_test; C2_test]);
    end
    R0_test = setdiff(1:length(group_test), union(R1_test, R2_test));
    
    fprintf('  测试集分类结果: R1=%d, R2=%d, R0=%d (C1=%d, C2=%d)\n', ...
        length(R1_test), length(R2_test), length(R0_test), length(C1_test), length(C2_test));

    % ============================================================
    %  计算测试集性能指标
    % ============================================================
    % 使用分类结果计算性能
    y_true_test = group_test(:);
    y_pred_test = zeros(length(group_test), 1);
    y_pred_test(R1_test) = 0;  % R1 对应原始标签 0
    y_pred_test(R2_test) = 1;  % R2 对应原始标签 1
    
    % 二分类性能指标
    y_true_bin = (y_true_test == 1);
    y_pred_bin = (y_pred_test == 1);
    
    TP = sum(y_true_bin == 1 & y_pred_bin == 1);
    FP = sum(y_true_bin == 0 & y_pred_bin == 1);
    FN = sum(y_true_bin == 1 & y_pred_bin == 0);
    TN = sum(y_true_bin == 0 & y_pred_bin == 0);
    all_TP(end+1) = TP;
    all_FP(end+1) = FP;
    all_FN(end+1) = FN;
    all_TN(end+1) = TN;
    file_TP(end+1) = TP;
    file_FP(end+1) = FP;
    file_FN(end+1) = FN;
    file_TN(end+1) = TN;
    best_test_precision = TP / (TP + FP + eps);
    best_test_recall = TP / (TP + FN + eps);
    best_test_F1 = 2 * best_test_precision * best_test_recall / (best_test_precision + best_test_recall + eps);
    best_test_accuracy = (TP + TN) / (TP + TN + FP + FN + eps);
    
    % AUC 计算（使用 -scores 使得正类得分高于负类得分）
    scores_auc = -test_scores;
    ranks = tiedrank(scores_auc);
    pos_idx = find(y_true_bin == 1);
    neg_idx = find(y_true_bin == 0);
    m = length(pos_idx);
    n_pos = length(neg_idx);
    S = sum(ranks(pos_idx));
    best_test_AUC = (S - m*(m+1)/2) / (m * n_pos + eps);

    fprintf('测试集性能：F1=%.4f, AUC=%.4f, Precision=%.4f, Recall=%.4f, Accuracy=%.4f\n', ...
        best_test_F1, best_test_AUC, best_test_precision, best_test_recall, best_test_accuracy);
    % ============================================================
    %  绘制并保存测试集混淆矩阵（配色参考上传图片）
    % ============================================================
    cm = [TN, FP;
          FN, TP];

    fig_cm = figure('Color', 'w', 'Position', [100, 100, 920, 820]);

    imagesc(cm);
    axis equal;
    axis tight;

    % 自定义配色：浅灰蓝 -> 中灰蓝 -> 深海军蓝（参考上传图片）
    custom_blue = [
        0.93 0.94 0.96
        0.84 0.87 0.90
        0.72 0.78 0.84
        0.50 0.63 0.77
        0.24 0.48 0.73
        0.06 0.23 0.49
    ];
    colormap(interp1(linspace(0,1,size(custom_blue,1)), custom_blue, linspace(0,1,256)));

    cb = colorbar;
    cb.FontName = 'Times New Roman';
    cb.FontSize = FONT_TEXT;
    cb.Color = [0.25 0.25 0.25];

    ax = gca;
    ax.FontName = 'Times New Roman';
    ax.FontSize = FONT_TICK;
    ax.LineWidth = 0.8;
    ax.XTick = [1 2];
    ax.YTick = [1 2];
    ax.XTickLabel = {'0', '1'};
    ax.YTickLabel = {'0', '1'};
    ax.TickLength = [0 0];
    ax.Box = 'off';
    ax.XColor = [0.30 0.30 0.30];
    ax.YColor = [0.30 0.30 0.30];

    xlabel('Predicted label', 'FontName', 'Times New Roman', 'FontSize', FONT_LABEL, 'Color', [0.25 0.25 0.25]);
    ylabel('True label', 'FontName', 'Times New Roman', 'FontSize', FONT_LABEL, 'Color', [0.25 0.25 0.25]);
    title('Confusion Matrix', 'FontName', 'Times New Roman', 'FontSize', FONT_TITLE, 'FontWeight', 'normal', 'Color', [0.10 0.10 0.10]);

    % 白色分隔线，模拟示例图中的方格感
    hold on;
    xline(1.5, 'w-', 'LineWidth', 0.8, 'Alpha', 0.85);
    yline(1.5, 'w-', 'LineWidth', 0.8, 'Alpha', 0.85);

    % 数值标注
    max_val = max(cm(:));
    for i = 1:size(cm,1)
        for j = 1:size(cm,2)
            value = cm(i,j);

            % 深色格子白字，浅色格子深蓝字
            if value >= 0.55 * max_val
                txtColor = [1 1 1];
            else
                txtColor = [0.06 0.23 0.49];
            end

            text(j, i, sprintf('%d', value), ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', ...
                'FontName', 'Times New Roman', ...
                'FontSize', FONT_TEXT, ...
                'FontWeight', 'normal', ...
                'Color', txtColor);
        end
    end
    hold off;

    % 保存
    saveas(fig_cm, fullfile(resultFolder, baseName + "_confusion_matrix_test.png"));
    close(fig_cm);

    fprintf('  测试集混淆矩阵已保存：%s\n', ...
        fullfile(resultFolder, baseName + "_confusion_matrix_test.png"));
    % ============================================================
    %  计算第一阶段测试集性能指标（仅使用R1和R2分类，不使用c对R0进一步分类）
    % ============================================================
    % 第一阶段：只考虑R1和R2，R0作为不确定区域不参与指标计算
    % R1: score > d → 预测为0（原始标签0）
    % R2: score < d - eta → 预测为1（原始标签1）
    % R0: d - eta <= score <= d → 不确定，不计入指标计算
    
    % 获取第一阶段的分类结果（仅R1和R2，不包含C1/C2）
    R1_test_stage1_only = find(test_scores > d_train);
    R2_test_stage1_only = find(test_scores < d_train - eta_train);

    % 计算第一阶段性能指标（仅使用R1和R2样本）
    y_true_stage1 = group_test(:);
    y_pred_stage1 = zeros(length(group_test), 1);
    y_pred_stage1(R1_test_stage1_only) = 0;  % R1 对应原始标签 0
    y_pred_stage1(R2_test_stage1_only) = 1;  % R2 对应原始标签 1
    
    % 仅使用R1和R2样本来计算指标（R0样本不计入）
    classified_idx_stage1 = [R1_test_stage1_only; R2_test_stage1_only];
    
    if ~isempty(classified_idx_stage1)
        y_true_stage1_classified = y_true_stage1(classified_idx_stage1);
        y_pred_stage1_classified = y_pred_stage1(classified_idx_stage1);
        
        y_true_bin_stage1 = (y_true_stage1_classified == 1);
        y_pred_bin_stage1 = (y_pred_stage1_classified == 1);
        
        TP_s1 = sum(y_true_bin_stage1 == 1 & y_pred_bin_stage1 == 1);
        FP_s1 = sum(y_true_bin_stage1 == 0 & y_pred_bin_stage1 == 1);
        FN_s1 = sum(y_true_bin_stage1 == 1 & y_pred_bin_stage1 == 0);
        TN_s1 = sum(y_true_bin_stage1 == 0 & y_pred_bin_stage1 == 0);
        
        best_test_precision_stage1 = TP_s1 / (TP_s1 + FP_s1 + eps);
        best_test_recall_stage1 = TP_s1 / (TP_s1 + FN_s1 + eps);
        best_test_F1_stage1 = 2 * best_test_precision_stage1 * best_test_recall_stage1 / (best_test_precision_stage1 + best_test_recall_stage1 + eps);
        best_test_accuracy_stage1 = (TP_s1 + TN_s1) / (TP_s1 + TN_s1 + FP_s1 + FN_s1 + eps);
        
        % AUC计算（仅使用R1和R2样本）
        if (TP_s1 + FN_s1) > 0 && (TN_s1 + FP_s1) > 0
            scores_stage1_classified = test_scores(classified_idx_stage1);
            scores_auc_s1 = -scores_stage1_classified;
            ranks_s1 = tiedrank(scores_auc_s1);
            pos_idx_s1 = find(y_true_bin_stage1 == 1);
            neg_idx_s1 = find(y_true_bin_stage1 == 0);
            m_s1 = length(pos_idx_s1);
            n_neg_s1 = length(neg_idx_s1);
            S_s1 = sum(ranks_s1(pos_idx_s1));
            best_test_AUC_stage1 = (S_s1 - m_s1*(m_s1+1)/2) / (m_s1 * n_neg_s1 + eps);
        else
            best_test_AUC_stage1 = NaN;
        end
    else
        best_test_precision_stage1 = NaN;
        best_test_recall_stage1 = NaN;
        best_test_F1_stage1 = NaN;
        best_test_accuracy_stage1 = NaN;
        best_test_AUC_stage1 = NaN;
    end
    
    fprintf('  第一阶段测试集性能（R1/R2 only）: F1=%.4f, AUC=%.4f, Precision=%.4f, Recall=%.4f, Accuracy=%.4f\n', ...
        best_test_F1_stage1, best_test_AUC_stage1, best_test_precision_stage1, best_test_recall_stage1, best_test_accuracy_stage1);

    % ============================================================
    %  计算第二阶段测试集性能指标（仅针对C1/C2，即R0中使用c细分的样本）
    % ============================================================
    % 第二阶段：仅使用R0区域中通过参数c细分为C1和C2的样本
    % C1: score > c → 预测为0（原始标签0）
    % C2: score <= c → 预测为1（原始标签1）
    
    if ~isempty(R0_candidates)
        % C1: R0中score > c，归属R1（原始标签0）
        % C2: R0中score <= c，归属R2（原始标签1）
        
        y_true_stage2 = group_test(:);
        y_pred_stage2 = zeros(length(group_test), 1);
        y_pred_stage2(C1_test) = 0;  % C1 对应原始标签 0
        y_pred_stage2(C2_test) = 1;  % C2 对应原始标签 1
        
        % 仅使用C1和C2样本来计算指标
        classified_idx_stage2 = [C1_test; C2_test];
        
        if ~isempty(classified_idx_stage2)
            y_true_stage2_classified = y_true_stage2(classified_idx_stage2);
            y_pred_stage2_classified = y_pred_stage2(classified_idx_stage2);
            
            y_true_bin_stage2 = (y_true_stage2_classified == 1);
            y_pred_bin_stage2 = (y_pred_stage2_classified == 1);
            
            TP_s2 = sum(y_true_bin_stage2 == 1 & y_pred_bin_stage2 == 1);
            FP_s2 = sum(y_true_bin_stage2 == 0 & y_pred_bin_stage2 == 1);
            FN_s2 = sum(y_true_bin_stage2 == 1 & y_pred_bin_stage2 == 0);
            TN_s2 = sum(y_true_bin_stage2 == 0 & y_pred_bin_stage2 == 0);
            
            if (TP_s2 + FP_s2) > 0
                best_test_precision_stage2 = TP_s2 / (TP_s2 + FP_s2 + eps);
            else
                best_test_precision_stage2 = NaN;
            end
            
            if (TP_s2 + FN_s2) > 0
                best_test_recall_stage2 = TP_s2 / (TP_s2 + FN_s2 + eps);
            else
                best_test_recall_stage2 = NaN;
            end
            
            best_test_F1_stage2 = 2 * best_test_precision_stage2 * best_test_recall_stage2 / (best_test_precision_stage2 + best_test_recall_stage2 + eps);
            best_test_accuracy_stage2 = (TP_s2 + TN_s2) / (TP_s2 + TN_s2 + FP_s2 + FN_s2 + eps);
            
            % AUC计算（仅使用C1/C2样本）
            if (TP_s2 + FN_s2) > 0 && (TN_s2 + FP_s2) > 0
                scores_stage2_classified = test_scores(classified_idx_stage2);
                scores_auc_s2 = -scores_stage2_classified;
                ranks_s2 = tiedrank(scores_auc_s2);
                pos_idx_s2 = find(y_true_bin_stage2 == 1);
                neg_idx_s2 = find(y_true_bin_stage2 == 0);
                m_s2 = length(pos_idx_s2);
                n_neg_s2 = length(neg_idx_s2);
                S_s2 = sum(ranks_s2(pos_idx_s2));
                best_test_AUC_stage2 = (S_s2 - m_s2*(m_s2+1)/2) / (m_s2 * n_neg_s2 + eps);
            else
                best_test_AUC_stage2 = NaN;
            end
        else
            best_test_precision_stage2 = NaN;
            best_test_recall_stage2 = NaN;
            best_test_F1_stage2 = NaN;
            best_test_accuracy_stage2 = NaN;
            best_test_AUC_stage2 = NaN;
        end
    else
        best_test_precision_stage2 = NaN;
        best_test_recall_stage2 = NaN;
        best_test_F1_stage2 = NaN;
        best_test_accuracy_stage2 = NaN;
        best_test_AUC_stage2 = NaN;
    end
    
    fprintf('  第二阶段测试集性能（C1/C2 only）: F1=%.4f, AUC=%.4f, Precision=%.4f, Recall=%.4f, Accuracy=%.4f\n', ...
        best_test_F1_stage2, best_test_AUC_stage2, best_test_precision_stage2, best_test_recall_stage2, best_test_accuracy_stage2);
    
    % 保存结果到 sol_test 结构体（兼容后续代码）
    sol_test = sol_train_final;  % 复制训练集的结果结构
    sol_test.theta = theta_train;
    sol_test.d = d_train;
    sol_test.scores = test_scores;
    sol_test.R1 = R1_test;
    sol_test.R2 = R2_test;
    sol_test.R0 = R0_test;
    sol_test.c = c_train;
    sol_test.F1 = best_test_F1;
    sol_test.AUC = best_test_AUC;
    sol_test.precision = best_test_precision;
    sol_test.recall = best_test_recall;
    sol_test.Accuracy = best_test_accuracy;
        %% ============================================================
    %  整理“数据划分信息 / K折信息 / 最优组信息 / 对照实验数据”
    %  用于后续机器学习对照实验
    % ============================================================

    % -------- 全部特征名 --------
    all_feature_names = zdata2024.Properties.VariableNames(2:end);
    if isstring(all_feature_names)
        all_feature_names = cellstr(all_feature_names);
    elseif ~iscell(all_feature_names)
        all_feature_names = cellstr(string(all_feature_names));
    end

    % -------- 最优组特征名 --------
    best_group_feature_names = all_feature_names(best_group_cols);
    if isstring(best_group_feature_names)
        best_group_feature_names = cellstr(best_group_feature_names);
    elseif ~iscell(best_group_feature_names)
        best_group_feature_names = cellstr(string(best_group_feature_names));
    end

    % ============================================================
    %  1) 训练集 / 测试集划分信息
    % ============================================================
    split_info = struct();
    split_info.file_name = fileName;
    split_info.base_name = baseName;
    split_info.random_seed = randomSeed;
    split_info.train_ratio = 0.70;
    split_info.test_ratio = 0.30;

    split_info.n_total = n_total;
    split_info.n_train = length(train_idx);
    split_info.n_test = length(test_idx);

    split_info.train_idx = train_idx;
    split_info.test_idx = test_idx;

    split_info.train_idx_class0 = train_idx_class0;
    split_info.train_idx_class1 = train_idx_class1;
    split_info.test_idx_class0 = test_idx_class0;
    split_info.test_idx_class1 = test_idx_class1;

    split_info.y_all = table2array(zdata2024(:,1));
    split_info.y_train = table2array(zdata2024_train(:,1));
    split_info.y_test = table2array(zdata2024_test(:,1));

    split_info.X_all = table2array(zdata2024(:,2:end));
    split_info.X_train = table2array(zdata2024_train(:,2:end));
    split_info.X_test = table2array(zdata2024_test(:,2:end));

    split_info.feature_names = all_feature_names;

    % ============================================================
    %  2) K折交叉验证划分信息
    % ============================================================
    cv_info = struct();
    cv_info.K_fold = K_fold;
    cv_info.idx_class0_train = idx_class0_train;
    cv_info.idx_class1_train = idx_class1_train;
    cv_info.cv_idx_class0 = cv_idx_class0;
    cv_info.cv_idx_class1 = cv_idx_class1;

    cv_info.fold_val_idx = cell(K_fold, 1);
    cv_info.fold_train_idx = cell(K_fold, 1);

    for k_cv = 1:K_fold
        val_idx_k = sort([cv_idx_class0{k_cv}; cv_idx_class1{k_cv}]);

        train_idx_k = [];
        for kk_cv = 1:K_fold
            if kk_cv ~= k_cv
                train_idx_k = [train_idx_k; cv_idx_class0{kk_cv}; cv_idx_class1{kk_cv}];
            end
        end
        train_idx_k = sort(train_idx_k);

        cv_info.fold_val_idx{k_cv} = val_idx_k;
        cv_info.fold_train_idx{k_cv} = train_idx_k;
    end

    % ============================================================
    %  3) 最优组别特征信息
    % ============================================================
    best_group_info = struct();
    best_group_info.best_group = best_group;
    best_group_info.best_group_cols = best_group_cols(:)';   % 相对于原始特征矩阵X的列索引
    best_group_info.feature_names = best_group_feature_names;

    % 最优组对应的特征矩阵（全数据 / 训练集 / 测试集）
    best_group_info.X_all_best_group = table2array(zdata2024(:, best_group_cols + 1));
    best_group_info.X_train_best_group = Z_group_best;
    best_group_info.X_test_best_group = Z_test_group;

    % 参数信息
    best_group_info.beta = best_beta;
    best_group_info.eta_cv = best_eta;      % 交叉验证阶段记录的eta
    best_group_info.eta_train = eta_train;  % 最终训练阶段自适应eta
    best_group_info.d = d_train;
    best_group_info.c = c_train;

    % 第一阶段 / 第二阶段权重
    best_group_info.theta_stage1 = theta_train(:);
    best_group_info.theta_stage2 = theta_stage2(:);

    % 最终输出theta
    if isfield(sol_test, 'theta') && ~isempty(sol_test.theta)
        best_group_info.theta_final = sol_test.theta(:);
    else
        best_group_info.theta_final = theta_train(:);
    end

    % 测试集预测结果
    best_group_info.test_scores = test_scores(:);
    best_group_info.y_test_true = y_true_test(:);
    best_group_info.y_test_pred = y_pred_test(:);

    % 区域划分
    best_group_info.R1_test = R1_test(:);
    best_group_info.R2_test = R2_test(:);
    best_group_info.R0_test = R0_test(:);
    best_group_info.C1_test = C1_test(:);
    best_group_info.C2_test = C2_test(:);

    best_group_info.R1_test_stage1_only = R1_test_stage1_only(:);
    best_group_info.R2_test_stage1_only = R2_test_stage1_only(:);

    % 性能指标
    best_group_info.metrics = struct( ...
        'F1', best_test_F1, ...
        'AUC', best_test_AUC, ...
        'precision', best_test_precision, ...
        'recall', best_test_recall, ...
        'accuracy', best_test_accuracy, ...
        'F1_stage1', best_test_F1_stage1, ...
        'AUC_stage1', best_test_AUC_stage1, ...
        'precision_stage1', best_test_precision_stage1, ...
        'recall_stage1', best_test_recall_stage1, ...
        'accuracy_stage1', best_test_accuracy_stage1, ...
        'F1_stage2', best_test_F1_stage2, ...
        'AUC_stage2', best_test_AUC_stage2, ...
        'precision_stage2', best_test_precision_stage2, ...
        'recall_stage2', best_test_recall_stage2, ...
        'accuracy_stage2', best_test_accuracy_stage2);

    % ============================================================
    %  4) 机器学习对照实验专用标准数据入口
    % ============================================================
    benchmark_data = struct();
    benchmark_data.file_name = fileName;
    benchmark_data.base_name = baseName;
    benchmark_data.random_seed = randomSeed;

    benchmark_data.train_idx = train_idx;
    benchmark_data.test_idx = test_idx;

    benchmark_data.feature_names = all_feature_names;

    benchmark_data.best_group = best_group;
    benchmark_data.best_group_cols = best_group_cols(:)';
    benchmark_data.best_group_feature_names = best_group_feature_names;

    % 全特征数据
    benchmark_data.X_train_full = split_info.X_train;
    benchmark_data.X_test_full = split_info.X_test;

    % 最优组特征数据
    benchmark_data.X_train_best = best_group_info.X_train_best_group;
    benchmark_data.X_test_best = best_group_info.X_test_best_group;

    % 标签
    benchmark_data.y_train = split_info.y_train;
    benchmark_data.y_test = split_info.y_test;
    % ============================================================
    %  保存第一阶段的 R1、R2、R0 和 scores（用于可视化）
    % ============================================================
    scores_stage1 = test_scores;
    R1_test_stage1 = R1_test;
    R2_test_stage1 = R2_test;
    R0_test_stage1 = R0_test;

    % 计算PR/ROC曲线数据（用于AUPR和组合曲线）
    % 使用模型输出的原始 scores，确保与模型内部 AUC 计算基于同一排序
    if isfield(sol_test, 'scores')
        raw_scores = sol_test.scores(:);
    else
        Z_test_group_T = Z_test_group.';
        raw_scores = sol_test.theta.' * Z_test_group_T;
        raw_scores = raw_scores(:);
    end
    
    y_true_test = (group_test == 1);
    
    % 根据模型内部 AUC 的排序方向自动选择使用 raw_scores 还是 -raw_scores，
    % 使得“分数越大越可能为正类”，并保证与模型内部排序一致（不改变 sol_test.AUC 数值）
    pos_idx_tmp = find(y_true_test == 1);
    neg_idx_tmp = find(y_true_test == 0);
if ~isempty(pos_idx_tmp) && ~isempty(neg_idx_tmp)
    % 使用与 model8_1_7 中类似的 rank-based 公式估计两种方向下的 AUC
    ranks_raw = tiedrank(raw_scores);
    m_pos = length(pos_idx_tmp);
    n_neg = length(neg_idx_tmp);
    S_raw = sum(ranks_raw(pos_idx_tmp));
    auc_raw = (S_raw - m_pos*(m_pos+1)/2) / (m_pos * n_neg);
    
    ranks_neg = tiedrank(-raw_scores);
    S_neg = sum(ranks_neg(pos_idx_tmp));
    auc_neg = (S_neg - m_pos*(m_pos+1)/2) / (m_pos * n_neg);
    
    % 选择与模型给出的 best_test_AUC 更接近的方向
    if abs(auc_raw - best_test_AUC) <= abs(auc_neg - best_test_AUC)
        test_scores_for_curve = raw_scores;
    else
        test_scores_for_curve = -raw_scores;
    end
else
    % 极端情况：只有一种类别，保持原分数
    test_scores_for_curve = raw_scores;
end

% 现在 test_scores_for_curve 已经生成，可安全写入结构体
best_group_info.test_scores_for_curve = test_scores_for_curve(:);
benchmark_data.test_scores_for_curve = test_scores_for_curve(:);
    
    % 直接使用所有唯一分数作为阈值（不再限于最多 100 个），保证 ROC/PR 曲线精确对应模型排序
    thresholds = sort(unique(test_scores_for_curve), 'descend');
    
    precision_curve = zeros(length(thresholds), 1);
    recall_curve = zeros(length(thresholds), 1);
    fpr_curve = zeros(length(thresholds), 1);  % FPR for ROC curve
    tpr_curve = zeros(length(thresholds), 1);  % TPR for ROC curve (same as recall)
    
    % 计算真实标签的统计信息
    n_pos = sum(y_true_test);  % 正样本数
    n_neg = sum(~y_true_test);  % 负样本数
    
    for t = 1:length(thresholds)
        y_pred_thresh = (test_scores_for_curve >= thresholds(t));
        TP = sum(y_true_test & y_pred_thresh);
        FP = sum(~y_true_test & y_pred_thresh);
        FN = sum(y_true_test & ~y_pred_thresh);
        TN = sum(~y_true_test & ~y_pred_thresh);
        
        % PR曲线计算
        if (TP + FP) > 0
            precision_curve(t) = TP / (TP + FP);
        else
            precision_curve(t) = 1;
        end
        
        if (TP + FN) > 0
            recall_curve(t) = TP / (TP + FN);
        else
            recall_curve(t) = 0;
        end
        
        % ROC曲线计算
        if n_pos > 0
            tpr_curve(t) = TP / n_pos;  % TPR = Recall = TP / (TP + FN) = TP / n_pos
        else
            tpr_curve(t) = 0;
        end
        
        if n_neg > 0
            fpr_curve(t) = FP / n_neg;  % FPR = FP / (FP + TN) = FP / n_neg
        else
            fpr_curve(t) = 0;
        end
    end
    
    % PR曲线处理
    [recall_sorted, sort_idx] = sort(recall_curve, 'ascend');
    precision_sorted = precision_curve(sort_idx);
    [recall_unique, unique_idx] = unique(recall_sorted, 'last');
    precision_unique = precision_sorted(unique_idx);
    
    % ROC曲线处理（按FPR排序）
    [fpr_sorted, sort_idx_roc] = sort(fpr_curve, 'ascend');
    tpr_sorted = tpr_curve(sort_idx_roc);
    % 移除重复的FPR值（保留最大的TPR）
    [fpr_unique, unique_idx_roc] = unique(fpr_sorted, 'last');
    tpr_unique = tpr_sorted(unique_idx_roc);
    
    % 确保ROC曲线从(0,0)开始，到(1,1)结束
    if isempty(fpr_unique) || fpr_unique(1) > 0
        fpr_unique = [0; fpr_unique];
        tpr_unique = [0; tpr_unique];
    end
    if fpr_unique(end) < 1
        fpr_unique = [fpr_unique; 1];
        tpr_unique = [tpr_unique; 1];
    end
    
    % 计算AUPR（用于保存）
    if length(recall_unique) > 1
        AUPR = trapz(recall_unique, precision_unique);
    else
        AUPR = precision_unique(1);
    end
    
    % 当前CSV文件测试集样本数（用于后续加权汇总；等于测试集行数）
    n_test_samples = length(group_test);
    
    % 保存当前CSV文件的best_group对应的测试集性能数据（用于后续绘制组合曲线和加权汇总）
    all_best_test_F1(end+1) = best_test_F1;
    all_best_test_AUC(end+1) = best_test_AUC;
    all_best_test_precision(end+1) = best_test_precision;
    all_best_test_recall(end+1) = best_test_recall;
    all_best_test_accuracy(end+1) = best_test_accuracy;
    all_best_test_n_samples(end+1) = n_test_samples;
    all_best_test_PR_curves{end+1} = struct('recall', recall_unique, 'precision', precision_unique);
    % 计算 PR-AUC（使用梯形积分）
    if ~isempty(recall_unique) && length(recall_unique) > 1
        % 按recall升序排序以确保正确计算积分
        [recall_sorted, sort_idx] = sort(recall_unique, 'ascend');
        precision_sorted = precision_unique(sort_idx);
        pr_auc = trapz(recall_sorted, precision_sorted);
    else
        pr_auc = 0;
    end
    all_best_test_PR_AUC(end+1) = pr_auc;
    all_best_test_ROC_curves{end+1} = struct('fpr', fpr_unique, 'tpr', tpr_unique);
    % 保存当前CSV文件的best_group对应的交叉验证性能数据（用于后续绘制组合曲线）
    all_best_CV_AUC(end+1) = best_AUC;
    all_best_CV_AUC_std(end+1) = best_AUC_std;
    all_best_CV_F1(end+1) = best_F1;
    all_best_CV_F1_std(end+1) = best_F1_std;
    
    % 收集当前CSV文件的测试集个体级别数据（用于计算整体综合性能）
    % 使用反转后的score（使得正类得分越高，概率越大）
    % 注意：所有CSV文件共享相同的正类样本，仅负类样本按分组不同
    current_cluster_id = length(all_best_test_AUC);  % 当前CSV文件的cluster编号
    n_test_samples = length(test_scores_for_curve);
    
    % 获取测试集在原始数据中的索引（用于识别重复样本）
    % test_idx是相对于原始数据的索引，可以用来识别是否是同一个样本
    test_idx_relative = test_idx;  % 保存当前文件的测试集索引
    
    % 使用与 ROC/PR 曲线一致的分数（分数越大越可能为正类），保证后续整体指标计算与模型排序一致
    all_individual_scores = [all_individual_scores; test_scores_for_curve(:)];
    all_individual_labels = [all_individual_labels; double(y_true_test(:))];  % true_label (0/1)，转换为数值
    file_individual_scores = [file_individual_scores; test_scores_for_curve(:)];
    file_individual_labels = [file_individual_labels; double(y_true_test(:))];
    all_individual_cluster_ids = [all_individual_cluster_ids; repmat(current_cluster_id, n_test_samples, 1)];  % cluster_id
    % 保存样本的特征值（用于基于特征值识别重复样本）
    % 注意：Z_test是测试集的特征矩阵，需要保存完整特征用于去重
    if isempty(all_individual_features)
        all_individual_features = Z_test;  % 第一次，直接保存
    else
        all_individual_features = [all_individual_features; Z_test];  % 追加
    end
    % 生成sample_id（文件名_样本索引），同时保存原始数据索引用于去重
    for si = 1:n_test_samples
        all_individual_sample_ids{end+1} = sprintf('%s_sample%d', baseName, si);
    end
    % 保存原始数据索引和文件名，用于后续去重识别
    all_individual_original_indices = [all_individual_original_indices; test_idx_relative(:)];  % 原始数据中的行索引
    for si = 1:n_test_samples
        all_individual_file_names{end+1} = fileName;  % 保存文件名
    end

    %% =====================================================
%   绘制第一阶段与第二阶段特征权重绝对值的组合雷达图
%   仅展示Stage1与Stage2相差最大的前8个特征
%   且按第一阶段 |theta| 顺时针降序排列
    % =====================================================

% -------- 获取 theta --------
theta_stage1 = theta_train(:);
theta_stage2 = theta_train;  % 默认

if isfield(sol_train_final, 'submodel') && isfield(sol_train_final.submodel, 'theta')
    theta_stage2 = sol_train_final.submodel.theta(:);
end

if false && length(theta_stage1) == length(theta_stage2) && ~isempty(theta_stage1)

    % -------- 获取变量名 --------
        if best_group <= numel(groups_unique)
        best_group_cols = groups_unique{best_group};
        if exist('original_varNames', 'var') && numel(original_varNames) >= max(best_group_cols)
            radar_var_names = original_varNames(best_group_cols);
            else
            radar_var_names = arrayfun(@(k) sprintf('Var%d', k), ...
                1:numel(theta_stage1), 'UniformOutput', false);
            end
        else
        radar_var_names = arrayfun(@(k) sprintf('Var%d', k), ...
            1:numel(theta_stage1), 'UniformOutput', false);
    end

    if isstring(radar_var_names)
        radar_var_names = cellstr(radar_var_names);
        end

    % -------- 计算绝对值 --------
    abs_theta_stage1 = abs(theta_stage1);
    abs_theta_stage2 = abs(theta_stage2);

    % -------- 计算两阶段差异 --------
    theta_diff = abs(abs_theta_stage1 - abs_theta_stage2);

    % -------- 选择差异最大的前8个 --------
    n_top = min(8, length(theta_diff));
    [~, idx_diff_sorted] = sort(theta_diff, 'descend');
    top_idx = idx_diff_sorted(1:n_top);

    % -------- 按第一阶段 |theta| 降序重新排序 --------
    [~, idx_stage1_sorted] = sort(abs_theta_stage1(top_idx), 'descend');
    top_idx = top_idx(idx_stage1_sorted);

    % -------- 提取数据 --------
    radar_var_names_top = radar_var_names(top_idx);
    theta_stage1_top = abs_theta_stage1(top_idx);
    theta_stage2_top = abs_theta_stage2(top_idx);

    % -------- 归一化（全局最大值）--------
    global_max = max([abs_theta_stage1; abs_theta_stage2]);
    if global_max > 0
        theta_stage1_norm = theta_stage1_top / global_max;
        theta_stage2_norm = theta_stage2_top / global_max;
    else
        theta_stage1_norm = theta_stage1_top;
        theta_stage2_norm = theta_stage2_top;
    end

    % -------- 构造闭合 --------
    theta_stage1_closed = [theta_stage1_norm; theta_stage1_norm(1)];
    theta_stage2_closed = [theta_stage2_norm; theta_stage2_norm(1)];

    n_vars = n_top;
    angles = linspace(0, 2*pi, n_vars+1);

    % -------- 绘图 --------
    fig = figure('Color','w');
    ax = polaraxes;
    hold(ax,'on');

    % Stage1
    polarplot(ax, angles, theta_stage1_closed, ...
        'Color', [0.23 0.47 0.84], 'LineWidth',2, 'DisplayName','Stage 1 (|θ₁|)');

    % Stage2
    polarplot(ax, angles, theta_stage2_closed, ...
        'Color', [0.75 0.20 0.20], 'LineWidth',2, 'DisplayName','Stage 2 (|θ₂|)');

    % -------- 坐标设置 --------
    ax.ThetaDir = 'clockwise';
    ax.ThetaZeroLocation = 'top';

    % 只设置 n_vars 个刻度（不要+1）
    ax.ThetaTick = angles(1:end-1) * 180/pi;
    ax.ThetaTickLabel = radar_var_names_top;
    ax.TickLabelInterpreter = 'none';
lgd = legend(ax);
lgd.Location = 'northeast';   % 先设一个基准
lgd.Units = 'normalized';
lgd.Position = [0.75 0.85 0.2 0.1];  % [left bottom width height]

    % -------- 保存 --------
    saveas(fig, fullfile(resultFolder, baseName + "_theta_radar_test.png"));
        close(fig);

    fprintf('  雷达图已保存（%d 个特征，按Stage1降序顺时针排列）\n', n_vars);
    end

    %% =====================================================
    %   写入最优参数总结表（保存到 resultFolder）
    % =====================================================
T = table( ...
    string(fileName), best_group, best_beta, best_eta, ...
    best_F1, best_AUC, best_precision, best_recall, best_accuracy, best_CV_TP, best_CV_FP, best_CV_FN, best_CV_TN, best_score, ...
    best_train_F1, best_train_AUC, best_train_precision, best_train_recall, best_train_accuracy, TP_train, FP_train, FN_train, TN_train, ...
    best_test_F1, best_test_AUC, best_test_precision, best_test_recall, best_test_accuracy, ...
    best_test_F1_stage1, best_test_AUC_stage1, best_test_precision_stage1, best_test_recall_stage1, best_test_accuracy_stage1, ...
    best_test_F1_stage2, best_test_AUC_stage2, best_test_precision_stage2, best_test_recall_stage2, best_test_accuracy_stage2 ...
    );

writetable(T, summaryFile, ...
    'WriteMode','append', ...
    'FileType','text');

% cluster级beta宽表已在前文写入：
%   rows = Metric，columns = Beta_01...Beta_09
% 为避免覆盖beta位点结果，此处不再写入旧版单列Metric-Value格式。


    %% =====================================================
    %   绘制第一阶段最优模板 R1、R2、R0 的 score 密度曲线
    % =====================================================
    % 使用第一阶段的结果（R1_test_stage1, R2_test_stage1, scores_stage1）
    if false && (~isempty(R1_test_stage1) || ~isempty(R2_test_stage1))
        % 确保 score 是列向量
        test_scores = scores_stage1(:);
        
        % 获取 R1、R2 对应的 score
        if ~isempty(R1_test_stage1)
            scores_R1 = test_scores(R1_test_stage1);
        else
            scores_R1 = [];
        end
        if ~isempty(R2_test_stage1)
            scores_R2 = test_scores(R2_test_stage1);
        else
            scores_R2 = [];
        end
        
        % 绘制 Score 密度曲线（核密度估计）
        fig = figure('Color', 'w');
        
        % 设置颜色
        color_R1 = [0.2 0.6 0.3];   % 绿色 - R1
        color_R2 = [0.8 0.2 0.3];    % 红色 - R2
        
        % 计算并绘制核密度估计曲线
        if ~isempty(scores_R1) && length(scores_R1) > 1
            [f_R1, xi_R1] = ksdensity(scores_R1);
            h1 = plot(xi_R1, f_R1, '-', 'LineWidth', 2.5, 'Color', color_R1, ...
                'DisplayName', sprintf('R1 (n=%d)', length(scores_R1)));
            hold on;
        end
        
        if ~isempty(scores_R2) && length(scores_R2) > 1
            [f_R2, xi_R2] = ksdensity(scores_R2);
            h2 = plot(xi_R2, f_R2, '-', 'LineWidth', 2.5, 'Color', color_R2, ...
                'DisplayName', sprintf('R2 (n=%d)', length(scores_R2)));
            hold on;
        end
        
        % 如果样本数太少，用散点图显示
        if ~isempty(scores_R1) && length(scores_R1) <= 1
            h1 = plot(scores_R1, 0, 'o', 'MarkerSize', 10, 'MarkerFaceColor', color_R1, ...
                'DisplayName', sprintf('R1 (n=%d)', length(scores_R1)));
            hold on;
        end
        if ~isempty(scores_R2) && length(scores_R2) <= 1
            h2 = plot(scores_R2, 0, 'o', 'MarkerSize', 10, 'MarkerFaceColor', color_R2, ...
                'DisplayName', sprintf('R2 (n=%d)', length(scores_R2)));
            hold on;
        end
        
        ax = gca;
        ax.GridColor = [0.88 0.88 0.88];
        ax.GridAlpha = 1;
        ax.FontName = 'Times New Roman';
        grid(ax, 'on');
        xlabel(ax, 'Score', 'FontName', 'Times New Roman', 'FontSize', FONT_LABEL);
        ylabel(ax, 'Density', 'FontName', 'Times New Roman', 'FontSize', FONT_LABEL);
        legend(ax, 'Location', 'best', 'FontName', 'Times New Roman', 'FontSize', FONT_LEGEND);
        
        % 保存图像
        saveas(fig, fullfile(resultFolder, [baseName, '_score_density_R1R2.png']));
        close(fig);
        
        fprintf('  R1/R2 Score density curve saved\n');
    end
    
    %% =====================================================
    %   绘制测试集PCA聚类可视化（第二阶段合并后的R1/R2聚类结果）
    % =====================================================
    % 使用Z_test_group的特征进行PCA降维
    if false && ~isempty(Z_test_group) && size(Z_test_group, 2) >= 2
        try
            % 执行PCA降维到2维
            [coeff, score, ~, ~, explained] = pca(Z_test_group);
            
            % 获取前两个主成分
            PC1 = score(:, 1);
            PC2 = score(:, 2);
            
            % 使用第二阶段合并后的R1和R2（已包含C1和C2）
            all_R1 = R1_test;  % 包含C1
            all_R2 = R2_test;  % 包含C2
            
            % 创建聚类标签：1=R1, 2=R2
            cluster_labels = zeros(size(PC1));
            cluster_labels(all_R1) = 1;
            cluster_labels(all_R2) = 2;
            
            fig = figure('Position', [100, 100, 429, 357], 'Color', 'w');
            ax = gca;
            hold(ax, 'on');
            
            % 绘制R2（负类，确定性预测为1）
            if ~isempty(all_R2)
                scatter(ax, PC1(all_R2), PC2(all_R2), 30, [0.85 0.2 0.2], 'filled', ...
                    'DisplayName', sprintf('R2 (n=%d)', length(all_R2)), ...
                    'MarkerFaceAlpha', 0.7);
            end
            
            % 绘制R1（正类，确定性预测为0）
            if ~isempty(all_R1)
                scatter(ax, PC1(all_R1), PC2(all_R1), 30, [0.2 0.5 0.85], 'filled', ...
                    'DisplayName', sprintf('R1 (n=%d)', length(all_R1)), ...
                    'MarkerFaceAlpha', 0.7);
            end
            
            ax.FontName = 'Times New Roman';
            grid(ax, 'on');
            ax.GridColor = [0.88 0.88 0.88];
            
            xlabel(ax, sprintf('PC1 (%.1f%%)', explained(1)), 'FontName', 'Times New Roman', 'FontSize', FONT_LABEL);
            ylabel(ax, sprintf('PC2 (%.1f%%)', explained(2)), 'FontName', 'Times New Roman', 'FontSize', FONT_LABEL);
            
            legend(ax, 'Location', 'best', 'FontName', 'Times New Roman', 'FontSize', FONT_LEGEND);
            
            saveas(fig, fullfile(resultFolder, [baseName, '_PCA_cluster.png']));
            close(fig);
            
            fprintf('  PCA聚类可视化已保存\n');
        catch pca_err
            fprintf('  警告：PCA可视化失败: %s\n', pca_err.message);
        end
    end
    
    %% ==== 保存当前文件的best_group对应的指标和权重 ====
    if best_group <= numel(groups_unique)
        best_group_cols = groups_unique{best_group};
        best_group_var_names = varNames(best_group_cols);
        % 确保转换为cell数组
        if isstring(best_group_var_names) || ischar(best_group_var_names)
            best_group_var_names = cellstr(best_group_var_names);
        elseif iscell(best_group_var_names)
            % 已经是cell数组，保持不变
        else
            best_group_var_names = cellstr(string(best_group_var_names));
        end
        all_best_group_vars{end+1} = best_group_var_names;
        
        % 保存theta权重（使用测试集上的theta）
        if isfield(sol_test, 'theta') && ~isempty(sol_test.theta)
            all_best_group_theta{end+1} = sol_test.theta(:);
        else
            all_best_group_theta{end+1} = [];
        end
        
        % 保存指标索引
        all_best_group_var_indices{end+1} = best_group_cols;
    else
        all_best_group_vars{end+1} = {};
        all_best_group_theta{end+1} = [];
        all_best_group_var_indices{end+1} = [];
    end
    
    % 保存文件名（不含扩展名）
    all_file_names{end+1} = erase(fileName, '.csv');
    
    % 保存完整的varNames（用于后续获取指标名称）
    all_var_names_full{end+1} = varNames;
    end

    %% =====================================================
    %   保存当前 synthetic_CTGAN 文件的 micro-level 汇总指标
    % =====================================================
    if ~isempty(file_TP)
        TP_total = sum(file_TP);
        FP_total = sum(file_FP);
        FN_total = sum(file_FN);
        TN_total = sum(file_TN);

        precision_total = TP_total / (TP_total + FP_total + eps);
        recall_total    = TP_total / (TP_total + FN_total + eps);
        F1_total        = 2 * precision_total * recall_total / (precision_total + recall_total + eps);
        accuracy_total  = (TP_total + TN_total) / (TP_total + TN_total + FP_total + FN_total + eps);

        if ~isempty(file_individual_scores)
            scores = file_individual_scores(:);
            labels = file_individual_labels(:);
            pos = labels == 1;
            neg = labels == 0;

            if sum(pos) > 0 && sum(neg) > 0
                ranks = tiedrank(scores);
                m = sum(pos);
                n = sum(neg);
                S = sum(ranks(pos));
                AUC_total = (S - m*(m+1)/2) / (m*n + eps);
            else
                AUC_total = NaN;
            end
        else
            AUC_total = NaN;
        end

        % 文件级训练集micro汇总：合并当前synthetic_CTGAN文件内所有有效cluster的完整训练集结果
        if ~isempty(file_train_TP)
            TP_train_total = sum(file_train_TP);
            FP_train_total = sum(file_train_FP);
            FN_train_total = sum(file_train_FN);
            TN_train_total = sum(file_train_TN);

            best_train_precision_total = TP_train_total / (TP_train_total + FP_train_total + eps);
            best_train_recall_total    = TP_train_total / (TP_train_total + FN_train_total + eps);
            best_train_F1_total        = 2 * best_train_precision_total * best_train_recall_total / (best_train_precision_total + best_train_recall_total + eps);
            best_train_accuracy_total  = (TP_train_total + TN_train_total) / (TP_train_total + TN_train_total + FP_train_total + FN_train_total + eps);

            if ~isempty(file_train_individual_scores)
                train_scores_file = file_train_individual_scores(:);
                train_labels_file = file_train_individual_labels(:);
                train_pos_file = train_labels_file == 1;
                train_neg_file = train_labels_file == 0;

                if sum(train_pos_file) > 0 && sum(train_neg_file) > 0
                    train_ranks_file = tiedrank(train_scores_file);
                    train_m_file = sum(train_pos_file);
                    train_n_file = sum(train_neg_file);
                    train_S_file = sum(train_ranks_file(train_pos_file));
                    best_train_AUC_total = (train_S_file - train_m_file*(train_m_file+1)/2) / (train_m_file * train_n_file + eps);
                else
                    best_train_AUC_total = NaN;
                end
            else
                best_train_AUC_total = NaN;
            end
        else
            TP_train_total = NaN;
            FP_train_total = NaN;
            FN_train_total = NaN;
            TN_train_total = NaN;
            best_train_precision_total = NaN;
            best_train_recall_total = NaN;
            best_train_F1_total = NaN;
            best_train_accuracy_total = NaN;
            best_train_AUC_total = NaN;
        end

        % 文件级交叉验证micro汇总：合并当前synthetic_CTGAN文件内所有有效cluster的CV验证折结果
        if ~isempty(file_cv_TP)
            TP_CV_total = sum(file_cv_TP);
            FP_CV_total = sum(file_cv_FP);
            FN_CV_total = sum(file_cv_FN);
            TN_CV_total = sum(file_cv_TN);

            best_CV_precision_total = TP_CV_total / (TP_CV_total + FP_CV_total + eps);
            best_CV_recall_total    = TP_CV_total / (TP_CV_total + FN_CV_total + eps);
            best_CV_F1_total        = 2 * best_CV_precision_total * best_CV_recall_total / (best_CV_precision_total + best_CV_recall_total + eps);
            best_CV_accuracy_total  = (TP_CV_total + TN_CV_total) / (TP_CV_total + TN_CV_total + FP_CV_total + FN_CV_total + eps);

            if ~isempty(file_cv_individual_scores)
                cv_scores_file = file_cv_individual_scores(:);
                cv_labels_file = file_cv_individual_labels(:);
                cv_pos_file = cv_labels_file == 1;
                cv_neg_file = cv_labels_file == 0;

                if sum(cv_pos_file) > 0 && sum(cv_neg_file) > 0
                    cv_ranks_file = tiedrank(cv_scores_file);
                    cv_m_file = sum(cv_pos_file);
                    cv_n_file = sum(cv_neg_file);
                    cv_S_file = sum(cv_ranks_file(cv_pos_file));
                    best_CV_AUC_total = (cv_S_file - cv_m_file*(cv_m_file+1)/2) / (cv_m_file * cv_n_file + eps);
                else
                    best_CV_AUC_total = NaN;
                end
            else
                best_CV_AUC_total = NaN;
            end
        else
            TP_CV_total = NaN;
            FP_CV_total = NaN;
            FN_CV_total = NaN;
            TN_CV_total = NaN;
            best_CV_precision_total = NaN;
            best_CV_recall_total = NaN;
            best_CV_F1_total = NaN;
            best_CV_accuracy_total = NaN;
            best_CV_AUC_total = NaN;
        end

        % =====================================================
        %   当前synthetic_CTGAN文件的beta位点级micro宽表
        %   每个beta位点合并该文件内所有有效cluster的TP/FP/FN/TN和score/label。
        % =====================================================
        betaFileMetricNames = {'best_CV_F1','best_CV_AUC','best_CV_precision','best_CV_recall','best_CV_accuracy', ...
                               'best_CV_TP','best_CV_FP','best_CV_FN','best_CV_TN', ...
                               'best_train_F1','best_train_AUC','best_train_precision','best_train_recall','best_train_accuracy', ...
                               'best_train_TP','best_train_FP','best_train_FN','best_train_TN', ...
                               'Precision','Recall','F1','Accuracy','AUC', ...
                               'TP','FP','FN','TN','beta'};
        fileBetaValues = NaN(numel(betaFileMetricNames), nBetaRequested);

        for beta_pos_file = 1:nBetaRequested
            [cv_p, cv_r, cv_f1, cv_acc] = binary_metrics_from_counts( ...
                file_beta_CV_TP(beta_pos_file), file_beta_CV_FP(beta_pos_file), file_beta_CV_FN(beta_pos_file), file_beta_CV_TN(beta_pos_file));
            cv_auc = rank_auc_local(file_beta_CV_scores{beta_pos_file}, file_beta_CV_labels{beta_pos_file});

            [tr_p, tr_r, tr_f1, tr_acc] = binary_metrics_from_counts( ...
                file_beta_train_TP(beta_pos_file), file_beta_train_FP(beta_pos_file), file_beta_train_FN(beta_pos_file), file_beta_train_TN(beta_pos_file));
            tr_auc = rank_auc_local(file_beta_train_scores{beta_pos_file}, file_beta_train_labels{beta_pos_file});

            [te_p, te_r, te_f1, te_acc] = binary_metrics_from_counts( ...
                file_beta_TP(beta_pos_file), file_beta_FP(beta_pos_file), file_beta_FN(beta_pos_file), file_beta_TN(beta_pos_file));
            te_auc = rank_auc_local(file_beta_scores{beta_pos_file}, file_beta_labels{beta_pos_file});

            if exist('betaGrid', 'var') && numel(betaGrid) >= beta_pos_file
                beta_value_file = betaGrid(beta_pos_file);
            else
                beta_value_file = NaN;
            end

            fileBetaValues(:, beta_pos_file) = [ ...
                cv_f1; cv_auc; cv_p; cv_r; cv_acc; ...
                file_beta_CV_TP(beta_pos_file); file_beta_CV_FP(beta_pos_file); file_beta_CV_FN(beta_pos_file); file_beta_CV_TN(beta_pos_file); ...
                tr_f1; tr_auc; tr_p; tr_r; tr_acc; ...
                file_beta_train_TP(beta_pos_file); file_beta_train_FP(beta_pos_file); file_beta_train_FN(beta_pos_file); file_beta_train_TN(beta_pos_file); ...
                te_p; te_r; te_f1; te_acc; te_auc; ...
                file_beta_TP(beta_pos_file); file_beta_FP(beta_pos_file); file_beta_FN(beta_pos_file); file_beta_TN(beta_pos_file); beta_value_file];
        end

        file_micro_metrics = fullfile(resultFolder, [originalBaseName, '_global_micro_metrics.csv']);
        write_beta_metric_table(file_micro_metrics, betaFileMetricNames, fileBetaValues, nBetaRequested);
        fprintf('当前文件beta位点micro指标已保存到：%s\n', file_micro_metrics);

        rowF1 = fileBetaValues(strcmp(betaFileMetricNames, 'F1'), :);
        rowRecall = fileBetaValues(strcmp(betaFileMetricNames, 'Recall'), :);
        all_beta_test_F1_by_file = [all_beta_test_F1_by_file; rowF1];
        all_beta_test_recall_by_file = [all_beta_test_recall_by_file; rowRecall];
    end
end

fprintf('\n全部文件处理完成！最优参数总结表已生成：%s\n', summaryFile);

end  % ~runVisualizationOnly：模型计算部分结束


%% =====================================================
%   计算每个 synthetic_CTGAN 文件 global_micro_metrics 的beta位点算术平均值和标准差
%   输出格式：除首列Metric外，Beta_01...Beta_09为各beta位点的跨文件平均性能。
% =====================================================
if runVisualizationOnly
    fprintf('仅可视化模式：从已有 synthetic_CTGAN_*_global_micro_metrics.csv 重新汇总并绘图。\n');
end

metricAverageFiles = dir(fullfile(resultFolder, 'synthetic_CTGAN_*_global_micro_metrics.csv'));

if ~isempty(metricAverageFiles)
    [meanWideTable, stdWideTable] = summarize_beta_metric_files(metricAverageFiles, nBetaRequested);

    metricAverageFile = fullfile(resultFolder, 'global_micro_metrics_arithmetic_mean.csv');
    writetable(meanWideTable, metricAverageFile, 'Encoding', 'GBK');

    metricStdFile = fullfile(resultFolder, 'global_micro_metrics_arithmetic_std.csv');
    writetable(stdWideTable, metricStdFile, 'Encoding', 'GBK');

    fprintf('global_micro_metrics按beta位点的算术平均值已保存到：%s\n', metricAverageFile);
    fprintf('global_micro_metrics按beta位点的标准差已保存到：%s\n', metricStdFile);
else
    fprintf('未找到 synthetic_CTGAN_*_global_micro_metrics.csv，跳过beta位点算术平均值计算。\n');
end

%% =====================================================
%   绘制beta位点CV性能曲线：仅展示best_CV_F1 / best_CV_recall
%   数据来源：global_micro_metrics_arithmetic_mean.csv 与
%             global_micro_metrics_arithmetic_std.csv
%   说明：本图不再依赖 all_beta_test_* 或 all_beta_CV_* 运行时变量。
% =====================================================
metricAverageFile = fullfile(resultFolder, 'global_micro_metrics_arithmetic_mean.csv');
metricStdFile     = fullfile(resultFolder, 'global_micro_metrics_arithmetic_std.csv');

if exist(metricAverageFile, 'file') && exist(metricStdFile, 'file')

    meanWideTable = readtable(metricAverageFile, 'VariableNamingRule', 'preserve', 'Encoding', 'GBK');
    stdWideTable  = readtable(metricStdFile,     'VariableNamingRule', 'preserve', 'Encoding', 'GBK');

    betaVarNames = arrayfun(@(i) sprintf('Beta_%02d', i), 1:nBetaRequested, 'UniformOutput', false);
    betaVarNames = betaVarNames(ismember(betaVarNames, meanWideTable.Properties.VariableNames));
    betaVarNames = betaVarNames(ismember(betaVarNames, stdWideTable.Properties.VariableNames));

    if isempty(betaVarNames)
        fprintf('global_micro_metrics_arithmetic_mean/std.csv中未找到Beta_01...Beta_%02d列，跳过beta位点CV曲线。\n', nBetaRequested);
    else
        beta_positions = 1:numel(betaVarNames);

        mean_CV_F1_beta     = get_metric_row_values(meanWideTable, 'best_CV_F1', betaVarNames);
        std_CV_F1_beta      = get_metric_row_values(stdWideTable,  'best_CV_F1', betaVarNames);
        mean_CV_recall_beta = get_metric_row_values(meanWideTable, 'best_CV_recall', betaVarNames);
        std_CV_recall_beta  = get_metric_row_values(stdWideTable,  'best_CV_recall', betaVarNames);

        has_CV_F1 = any(isfinite(mean_CV_F1_beta));
        has_CV_recall = any(isfinite(mean_CV_recall_beta));

        if ~has_CV_F1 && ~has_CV_recall
            fprintf('global_micro_metrics_arithmetic_mean.csv中未找到有效best_CV_F1或best_CV_recall，跳过beta位点CV曲线。\n');
        else
            fig_beta = figure('Color', 'w', 'Position', [100, 100, 1050, 620]);
            ax = axes(fig_beta);
            hold(ax, 'on');

            x_fill = [beta_positions, fliplr(beta_positions)];

            % ===== CV F1：均值 ± 标准差 =====
            if has_CV_F1
                f1_lower = mean_CV_F1_beta - std_CV_F1_beta;
                f1_upper = mean_CV_F1_beta + std_CV_F1_beta;
                fill(ax, x_fill, [f1_lower, fliplr(f1_upper)], [0.25 0.45 0.90], ...
                    'FaceAlpha', 0.18, 'EdgeColor', 'none', 'HandleVisibility', 'off');
                plot(ax, beta_positions, mean_CV_F1_beta, '--', ...
                    'LineWidth', 3.0, ...
                    'Color', [0.25 0.45 0.90], ...
                    'DisplayName', 'best\_CV\_F1');
            end

            % ===== CV Recall：均值 ± 标准差 =====
            if has_CV_recall
                recall_lower = mean_CV_recall_beta - std_CV_recall_beta;
                recall_upper = mean_CV_recall_beta + std_CV_recall_beta;
                fill(ax, x_fill, [recall_lower, fliplr(recall_upper)], [0.86 0.35 0.52], ...
                    'FaceAlpha', 0.22, 'EdgeColor', 'none', 'HandleVisibility', 'off');
                plot(ax, beta_positions, mean_CV_recall_beta, '--', ...
                    'LineWidth', 3.0, ...
                    'Color', [0.86 0.35 0.52], ...
                    'DisplayName', 'best\_CV\_recall');
            end

            ax.FontName = 'Times New Roman';
            ax.FontSize = FONT_TICK;
            ax.LineWidth = 1.1;
            ax.Box = 'on';
            ax.XGrid = 'on';
            ax.YGrid = 'on';
            ax.GridColor = [0.86 0.86 0.86];
            ax.GridAlpha = 1;
            ax.XLim = [1 numel(betaVarNames)];
            ax.XTick = beta_positions;

            % ===== y轴范围：仅考虑CV曲线及其标准差阴影 =====
            y_candidates = [];
            if has_CV_F1
                y_candidates = [y_candidates; f1_lower(:); f1_upper(:)]; %#ok<AGROW>
            end
            if has_CV_recall
                y_candidates = [y_candidates; recall_lower(:); recall_upper(:)]; %#ok<AGROW>
            end
            y_min = min(y_candidates, [], 'omitnan');
            y_max = max(y_candidates, [], 'omitnan');
            if isfinite(y_min) && isfinite(y_max) && y_max > y_min
                ylim(ax, [max(0, y_min - 0.02), min(1, y_max + 0.02)]);
            else
                ylim(ax, [0 1]);
            end

            xlabel(ax, 'Ordered beta positions', ...
                'FontName', 'Times New Roman', ...
                'FontSize', FONT_LABEL);

            % 横坐标突出 beta=1、本文 beta0 和 beta=g1/g2 三个理论锚点
            if exist('beta_tick_labels', 'var') && numel(beta_tick_labels) == numel(betaVarNames)
                ax.XTickLabel = beta_tick_labels;
            else
                ax.XTickLabel = {'1', '\beta_0', '', '', '', '', '', '', 'g_1/g_2'};
            end
            ax.TickLabelInterpreter = 'tex';

            ylabel(ax, 'Metric Score', ...
                'FontName', 'Times New Roman', ...
                'FontSize', FONT_LABEL);

            title(ax, 'Monte Carlo CV Performance of F1-score and Recall Across Beta Positions', ...
                'FontName', 'Times New Roman', ...
                'FontSize', FONT_TITLE, ...
                'FontWeight', 'bold');

            lgd = legend(ax, 'Location', 'northeast', 'Box', 'on');
            lgd.FontName = 'Times New Roman';
            lgd.FontSize = FONT_LEGEND;

            exportgraphics(fig_beta, ...
                fullfile(resultFolder, 'beta_position_F1_recall_performance.png'), ...
                'Resolution', 300);

            close(fig_beta);

            fprintf('beta位点CV F1/Recall性能曲线已保存：%s\n', ...
                fullfile(resultFolder, 'beta_position_F1_recall_performance.png'));
        end
    end
else
    fprintf('未找到global_micro_metrics_arithmetic_mean.csv或global_micro_metrics_arithmetic_std.csv，跳过beta位点CV曲线。\n');
end
if runVisualizationOnly
    fprintf('\n仅可视化部分运行完毕。\n');
end

%% =====================================================
%   基于每个CSV文件测试集结果的汇总指标
% =====================================================
if false && ~isempty(all_TP)

    fprintf('\n开始计算基于全局TP/FP/FN/TN的整体指标（micro-level）...\n');

    % =========================
    % 1. 全局混淆矩阵
    % =========================
    TP_total = sum(all_TP);
    FP_total = sum(all_FP);
    FN_total = sum(all_FN);
    TN_total = sum(all_TN);

    % =========================
    % 2. 重新计算指标（核心）
    % =========================
    precision_total = TP_total / (TP_total + FP_total + eps);
    recall_total    = TP_total / (TP_total + FN_total + eps);
    F1_total        = 2 * precision_total * recall_total / (precision_total + recall_total + eps);
    accuracy_total  = (TP_total + TN_total) / (TP_total + TN_total + FP_total + FN_total + eps);

    % =========================
    % 3. AUC（micro-level重新拼接法）
    % =========================
    % 更严谨做法：需要所有样本score+label（你其实已经有）
    % 用 all_individual_scores / labels 最准确

    if ~isempty(all_individual_scores)
        scores = all_individual_scores(:);
        labels = all_individual_labels(:);

        pos = labels == 1;
        neg = labels == 0;

        if sum(pos) > 0 && sum(neg) > 0
            ranks = tiedrank(-scores);
            m = sum(pos);
            n = sum(neg);
            S = sum(ranks(pos));
            AUC_total = (S - m*(m+1)/2) / (m*n + eps);
        else
            AUC_total = NaN;
        end
    else
        AUC_total = NaN;
    end

    % =========================
    % 4. 输出CSV
    % =========================
    weighted_metrics_file = fullfile(resultFolder, 'global_micro_metrics.csv');
    fid_w = fopen(weighted_metrics_file, 'w', 'n', 'GBK');

    if fid_w ~= -1

        fprintf(fid_w, 'Metric,Value\n');
        fprintf(fid_w, 'TP,%d\n', TP_total);
        fprintf(fid_w, 'FP,%d\n', FP_total);
        fprintf(fid_w, 'FN,%d\n', FN_total);
        fprintf(fid_w, 'TN,%d\n', TN_total);

        fprintf(fid_w, 'Precision,%.6f\n', precision_total);
        fprintf(fid_w, 'Recall,%.6f\n', recall_total);
        fprintf(fid_w, 'F1,%.6f\n', F1_total);
        fprintf(fid_w, 'Accuracy,%.6f\n', accuracy_total);
        fprintf(fid_w, 'AUC,%.6f\n', AUC_total);

        fclose(fid_w);

        fprintf('全局micro指标已保存到：%s\n', weighted_metrics_file);
    else
        fprintf('警告：无法写入文件\n');
    end
end

%% =====================================================
%   绘制所有CSV文件的best_group交叉验证平均性能F1柱状图（带方差线）
% =====================================================
if false && ~isempty(all_best_CV_F1) && length(all_best_CV_F1) > 0
    num_files = length(all_best_CV_F1);
    
    % 配色方案：深青色、紫色、蓝色、粉色
    colors = [0.13 0.59 0.95; 0.58 0.0 0.83; 0.0 0.68 1.0; 1.0 0.41 0.71];
    
    % 安全地生成颜色索引
    color_idx = mod(0:num_files-1, size(colors,1)) + 1;
    bar_colors = colors(color_idx, :);
    
    fig = figure('Position', [100, 100, 429, 286], 'Color', 'w');
        
    % 水平条形图
    y_pos = 1:num_files;
    
    h = barh(y_pos, all_best_CV_F1, 0.6, ...
        'EdgeColor', 'k', 'LineWidth', 0.5);
    
    % 关键修正
    h.FaceColor = 'flat';
    h.CData = bar_colors;
    
    hold on;
        
    % 误差线（黑色细线，较小的横向端点）
   errorbar( ...
    all_best_CV_F1, ...      % x
    y_pos, ...               % y
    all_best_CV_F1_std, ...  % 左误差
    all_best_CV_F1_std, ...  % 右误差
    'horizontal', ...
    'k', ...
    'LineWidth', 1, ...
    'LineStyle','none', ...
    'CapSize', 6);
    set(gca,'Layer','top');   % 防止误差线被遮挡
    ax = gca;
    ax.FontName = 'Times New Roman';
    ax.YDir = 'reverse';  % 使第一个在顶部
    
    % X轴范围 [0, 1]，刻度间隔 0.2
    xlim(ax, [0, 1]);
    ax.XTick = 0:0.2:1;
    
    % 不显示背景网格线，保留外框
    grid(ax, 'off');
    ax.Box = 'on';
    
    % 设置y轴标签
    ax.YTick = y_pos;
    ax.YTickLabel = arrayfun(@(x) sprintf('cluster%d', x), 1:num_files, 'UniformOutput', false);
    ax.TickLabelInterpreter = 'none';
    
    xlabel(ax, 'F1 Value', 'FontName', 'Times New Roman');
    ylabel(ax, 'CSV File', 'FontName', 'Times New Roman');
    
    % 添加数值标签
    for i = 1:num_files
        text(ax, all_best_CV_F1(i) + 0.02, y_pos(i), sprintf('%.3f', all_best_CV_F1(i)), ...
            'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'FontSize', FONT_TEXT, 'FontName', 'Times New Roman');
    end
    
    saveas(fig, fullfile(resultFolder, 'all_files_CV_F1_bar_with_std.png'));
    close(fig);
    
    fprintf('所有CSV文件的交叉验证平均性能F1柱状图已绘制\n');
end

% 绘制所有CSV文件的best_group交叉验证平均性能AUC柱状图（带方差线）
if false && ~isempty(all_best_CV_AUC) && length(all_best_CV_AUC) > 0
    num_files = length(all_best_CV_AUC);
    
    % 配色方案：深青色、紫色、蓝色、粉色
    colors = [0.13 0.59 0.95; 0.58 0.0 0.83; 0.0 0.68 1.0; 1.0 0.41 0.71];
    
    % 安全地生成颜色索引
    color_idx = mod(0:num_files-1, size(colors,1)) + 1;
    bar_colors = colors(color_idx, :);
    
    fig = figure('Position', [100, 100, 429, 286], 'Color', 'w');
    
    % 水平条形图
    y_pos = 1:num_files;
    
    h = barh(y_pos, all_best_CV_AUC, 0.6, ...
        'EdgeColor', 'k', 'LineWidth', 0.5);
    
    % 关键修正
    h.FaceColor = 'flat';
    h.CData = bar_colors;
    
    hold on;
    
    % 误差线（黑色细线，较小的横向端点）
    errorbar( ...
        all_best_CV_AUC, ...      % x
        y_pos, ...               % y
        all_best_CV_AUC_std, ...  % 左误差
        all_best_CV_AUC_std, ...  % 右误差
        'horizontal', ...
        'LineStyle','none', ...   % 不绘制连线
        'Color','k', ...          % 误差线颜色设置为黑色
        'LineWidth', 1, ...
        'CapSize', 6);
    
    set(gca,'Layer','top');   % 防止误差线被遮挡
    ax = gca;
    ax.FontName = 'Times New Roman';
    ax.YDir = 'reverse';  % 使第一个在顶部
    % X轴范围 [0, 1]，刻度间隔 0.2
    xlim(ax, [0, 1]);
    ax.XTick = 0:0.2:1;
    
    % 不显示背景网格线，保留外框
    grid(ax, 'off');
    ax.Box = 'on';
    
    % 设置y轴标签
    ax.YTick = y_pos;
    ax.YTickLabel = arrayfun(@(x) sprintf('cluster%d', x), 1:num_files, 'UniformOutput', false);
    ax.TickLabelInterpreter = 'none';
    
    xlabel(ax, 'AUC Value', 'FontName', 'Times New Roman');
    ylabel(ax, 'CSV File', 'FontName', 'Times New Roman');
    
    % 添加数值标签
    for i = 1:num_files
        text(ax, all_best_CV_AUC(i) + 0.02, y_pos(i), sprintf('%.3f', all_best_CV_AUC(i)), ...
            'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'FontSize', FONT_TEXT, 'FontName', 'Times New Roman');
    end
    
    saveas(fig, fullfile(resultFolder, 'all_files_CV_AUC_bar_with_std.png'));
    close(fig);
    
    fprintf('所有CSV文件的交叉验证平均性能AUC柱状图已绘制\n');
end

%% =====================================================
%   仅绘制 ROC 曲线并保存
% =====================================================
if false && ~isempty(all_best_test_ROC_curves) && length(all_best_test_ROC_curves) > 0
    colors = [0 0 0; 0.8 0.2 0.3; 0.2 0.5 0.85; 0.9 0.55 0.2; 0.35 0.25 0.6; 0.5 0.5 0.5];
    line_styles = {'-', '-', '-', '--', ':', '-.'};
    fig = figure('Position', [100, 100, 560, 520], 'Color', 'w');
    ax = gca;
    hold(ax, 'on');
    plot(ax, [0, 1], [0, 1], 'k--', 'LineWidth', 0.8, 'DisplayName', 'Random');
    for fi = 1:length(all_best_test_ROC_curves)
        if ~isempty(all_best_test_ROC_curves{fi}) && isfield(all_best_test_ROC_curves{fi}, 'fpr') && ~isempty(all_best_test_ROC_curves{fi}.fpr)
            roc_data = all_best_test_ROC_curves{fi};
            c = colors(mod(fi-1, size(colors,1))+1, :);
            ls = line_styles{mod(fi-1, length(line_styles))+1};
            plot(ax, roc_data.fpr, roc_data.tpr, ls, 'LineWidth', 1.2, 'Color', c, ...
                'DisplayName', sprintf('cluster%d (AUC=%.3f)', fi, all_best_test_AUC(fi)));
        end
    end
    ax.GridColor = [0.88 0.88 0.88];
    ax.GridAlpha = 1;
    ax.FontName = 'Times New Roman';
    grid(ax, 'on');
    xlabel(ax, 'False Positive Rate', 'FontName', 'Times New Roman');
    ylabel(ax, 'True Positive Rate', 'FontName', 'Times New Roman');
    xlim(ax, [0 1]); ylim(ax, [0 1]);
    axis(ax, 'square');
    legend(ax, 'Location', 'southeast', 'Interpreter', 'none', 'FontName', 'Times New Roman');
    saveas(fig, fullfile(resultFolder, 'all_files_test_ROC_combined.png'));
    close(fig);
    fprintf('ROC 曲线已单独保存：all_files_test_ROC_combined.png\n');
end

%% =====================================================
%   仅绘制 PR 曲线并保存
% =====================================================
if false && ~isempty(all_best_test_PR_curves) && length(all_best_test_PR_curves) > 0
    colors = [0 0 0; 0.8 0.2 0.3; 0.2 0.5 0.85; 0.9 0.55 0.2; 0.35 0.25 0.6; 0.5 0.5 0.5];
    line_styles = {'-', '-', '-', '--', ':', '-.'};
    fig = figure('Position', [100, 100, 560, 520], 'Color', 'w');
    ax = gca;
    hold(ax, 'on');
    for fi = 1:length(all_best_test_PR_curves)
        if ~isempty(all_best_test_PR_curves{fi}) && isfield(all_best_test_PR_curves{fi}, 'recall') && ~isempty(all_best_test_PR_curves{fi}.recall)
            pr_data = all_best_test_PR_curves{fi};
            c = colors(mod(fi-1, size(colors,1))+1, :);
            ls = line_styles{mod(fi-1, length(line_styles))+1};
            plot(ax, pr_data.recall, pr_data.precision, ls, 'LineWidth', 1.2, 'Color', c, ...
                'DisplayName', sprintf('cluster%d (PR-AUC=%.3f)', fi, all_best_test_PR_AUC(fi)));
        end
    end
    ax.GridColor = [0.88 0.88 0.88];
    ax.GridAlpha = 1;
    ax.FontName = 'Times New Roman';
    grid(ax, 'on');
    xlabel(ax, 'Recall', 'FontName', 'Times New Roman');
    ylabel(ax, 'Precision', 'FontName', 'Times New Roman');
    xlim(ax, [0 1]); ylim(ax, [0 1]);
    axis(ax, 'square');
    legend(ax, 'Location', 'southeast', 'Interpreter', 'none', 'FontName', 'Times New Roman');
    saveas(fig, fullfile(resultFolder, 'all_files_test_PR_combined.png'));
    close(fig);
    fprintf('PR 曲线已单独保存：all_files_test_PR_combined.png\n');
end


%% =====================================================
%   已去除 all_files_theta_heatmap 可视化。
%   本脚本改为输出 beta_position_F1_recall_performance.png。
% =====================================================

%% =====================================================
%   Local helper functions
%% =====================================================

function values = get_metric_row_values(T, metricName, betaVarNames)
    values = NaN(1, numel(betaVarNames));

    if ~ismember('Metric', T.Properties.VariableNames)
        return;
    end

    rowIdx = strcmp(string(T.Metric), metricName);
    if ~any(rowIdx)
        return;
    end

    rowPos = find(rowIdx, 1);
    for bi = 1:numel(betaVarNames)
        betaName = betaVarNames{bi};
        if ismember(betaName, T.Properties.VariableNames)
            values(bi) = double(T.(betaName)(rowPos));
        end
    end
end

function write_beta_metric_table(filePath, metricNames, metricValues, nBeta)
    betaVarNames = arrayfun(@(i) sprintf('Beta_%02d', i), 1:nBeta, 'UniformOutput', false);
    T = table(string(metricNames(:)), 'VariableNames', {'Metric'});
    for i = 1:nBeta
        if i <= size(metricValues, 2)
            T.(betaVarNames{i}) = metricValues(:, i);
        else
            T.(betaVarNames{i}) = NaN(numel(metricNames), 1);
        end
    end
    writetable(T, filePath, 'Encoding', 'GBK');
end

function [meanWideTable, stdWideTable] = summarize_beta_metric_files(metricFiles, nBeta)
    betaVarNames = arrayfun(@(i) sprintf('Beta_%02d', i), 1:nBeta, 'UniformOutput', false);

    metricNamesAll = {};
    fileTables = cell(numel(metricFiles), 1);

    for mf = 1:numel(metricFiles)
        T = readtable(fullfile(metricFiles(mf).folder, metricFiles(mf).name), 'VariableNamingRule', 'preserve');
        fileTables{mf} = T;
        metricNamesAll = [metricNamesAll; cellstr(string(T.Metric(:)))]; %#ok<AGROW>
    end

    metricNames = unique(metricNamesAll, 'stable');

    meanMat = NaN(numel(metricNames), nBeta);
    stdMat = NaN(numel(metricNames), nBeta);

    for mi = 1:numel(metricNames)
        metricName = metricNames{mi};

        for bi = 1:nBeta
            betaName = betaVarNames{bi};
            vals = NaN(numel(fileTables), 1);

            for mf = 1:numel(fileTables)
                T = fileTables{mf};
                if ~ismember(betaName, T.Properties.VariableNames)
                    continue;
                end
                rowIdx = strcmp(string(T.Metric), metricName);
                if any(rowIdx)
                    vals(mf) = double(T.(betaName)(find(rowIdx, 1)));
                end
            end

            meanMat(mi, bi) = mean(vals, 'omitnan');
            stdMat(mi, bi) = std(vals, 0, 'omitnan');
        end
    end

    meanWideTable = table(string(metricNames(:)), 'VariableNames', {'Metric'});
    stdWideTable = table(string(metricNames(:)), 'VariableNames', {'Metric'});
    for bi = 1:nBeta
        meanWideTable.(betaVarNames{bi}) = meanMat(:, bi);
        stdWideTable.(betaVarNames{bi}) = stdMat(:, bi);
    end
end

function [precision, recall, F1, accuracy] = binary_metrics_from_counts(TP, FP, FN, TN)
    precision = TP / (TP + FP + eps);
    recall = TP / (TP + FN + eps);
    F1 = 2 * precision * recall / (precision + recall + eps);
    accuracy = (TP + TN) / (TP + FP + FN + TN + eps);
end

function AUC = rank_auc_local(scores, labels)
    scores = scores(:);
    labels = labels(:);

    if isempty(scores) || isempty(labels)
        AUC = NaN;
        return;
    end

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

function evalOut = evaluate_dadea_final_model(Z_train_best, y_train, Z_test_best, y_test, beta, eta, group_cols, varNames, indicator_direction, opts)
    xi_input = zeros(1, size(Z_train_best, 2));

    for v = 1:length(group_cols)
        var_idx = group_cols(v);
        if var_idx <= length(varNames)
            var_name = varNames(var_idx);
            if iscell(var_name)
                var_name_str = char(var_name{1});
            elseif isstring(var_name)
                var_name_str = char(var_name);
            else
                var_name_str = char(string(var_name));
            end

            var_name_clean = strtrim(var_name_str);
            if isKey(indicator_direction, var_name_clean)
                xi_input(v) = indicator_direction(var_name_clean);
            end
        end
    end

    xi_p = (xi_input == 1) | (xi_input == 0);
    xi_m = (xi_input == -1) | (xi_input == 0);

    [R1_train, R2_train, ~, sol_train] = model8_1_7(Z_train_best, y_train, beta, eta, xi_p, xi_m, opts);

    theta_train = sol_train.theta;
    d_train = sol_train.d;
    eta_train = eta;

    if isfield(sol_train, 'submodel') && isfield(sol_train.submodel, 'c')
        c_train = sol_train.submodel.c;
    else
        c_train = d_train - eta_train / 2;
    end

    theta_stage2 = theta_train;
    if isfield(sol_train, 'submodel') && isfield(sol_train.submodel, 'theta')
        theta_stage2 = sol_train.submodel.theta;
    end

    scores_train = theta_train.' * Z_train_best.';
    scores_train = scores_train(:);

    y_pred_train = zeros(length(y_train), 1);
    y_pred_train(R1_train) = 0;
    y_pred_train(R2_train) = 1;
    trainMetrics = compute_metrics_with_scores(y_train(:), y_pred_train(:), -scores_train(:));

    test_scores = theta_train.' * Z_test_best.';
    test_scores = test_scores(:);

    R1_test = find(test_scores > d_train);
    R2_test = find(test_scores < d_train - eta_train);
    R0_candidates = find((test_scores >= d_train - eta_train) & (test_scores <= d_train));

    if ~isempty(R0_candidates)
        Z_R0_test = Z_test_best(R0_candidates, :);
        scores_R0_stage2 = theta_stage2.' * Z_R0_test.';
        scores_R0_stage2 = scores_R0_stage2(:);

        C1_test = R0_candidates(scores_R0_stage2 > c_train);
        C2_test = R0_candidates(scores_R0_stage2 <= c_train);

        R1_test = unique([R1_test; C1_test]);
        R2_test = unique([R2_test; C2_test]);
    end

    y_pred_test = zeros(length(y_test), 1);
    y_pred_test(R1_test) = 0;
    y_pred_test(R2_test) = 1;
    testMetrics = compute_metrics_with_scores(y_test(:), y_pred_test(:), -test_scores(:));

    evalOut = struct();
    evalOut.train = trainMetrics;
    evalOut.test = testMetrics;
end

function metrics = compute_metrics_with_scores(y_true, y_pred, scores)
    y_true = double(y_true(:));
    y_pred = double(y_pred(:));
    scores = double(scores(:));

    y_true_bin = (y_true == 1);
    y_pred_bin = (y_pred == 1);

    TP = sum(y_true_bin == 1 & y_pred_bin == 1);
    FP = sum(y_true_bin == 0 & y_pred_bin == 1);
    FN = sum(y_true_bin == 1 & y_pred_bin == 0);
    TN = sum(y_true_bin == 0 & y_pred_bin == 0);

    [precision, recall, F1, accuracy] = binary_metrics_from_counts(TP, FP, FN, TN);
    AUC = rank_auc_local(scores, double(y_true_bin));

    metrics = struct();
    metrics.TP = TP;
    metrics.FP = FP;
    metrics.FN = FN;
    metrics.TN = TN;
    metrics.precision = precision;
    metrics.recall = recall;
    metrics.F1 = F1;
    metrics.accuracy = accuracy;
    metrics.AUC = AUC;
    metrics.scores = scores(:);
    metrics.labels = double(y_true_bin(:));
end
