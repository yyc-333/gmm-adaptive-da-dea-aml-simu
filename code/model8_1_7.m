function [R1,R2,R0,sol] = model8_1_7(Z,group,beta,eta,xi_p,xi_m,opts) 
% model8_1_7 (theta - gamma 关联，xi 加权归一化)
% Z (n x k), group (1 x n 或 n x 1), beta, eta
% xi_p, xi_m: kx1 (或 1xk) 向量，表示 xi^+ 与 xi^-
% opts.vmin: theta 下界

% ========== 基本检查与信息 ==========
[n, k] = size(Z);
group = reshape(group,1,[]);
if numel(group) ~= n
    error('group 长度需等于样本数 n');
end

% 转列向量并校验 xi
xi_p = xi_p(:);
xi_m = xi_m(:);
if numel(xi_p) ~= k || numel(xi_m) ~= k
    error('xi_p 和 xi_m 长度必须均为 k（Z 的列数）');
end
if any(xi_p < 0) || any(xi_m < 0)
    error('xi_p 和 xi_m 必须为非负数（通常为 0/1 指示）');
end

Zc = Z.';  % k x n

% 样本分组
G1_idx = find(group==0);
G2_idx = find(group==1);
n1 = numel(G1_idx);
n2 = numel(G2_idx);

% ========= 变量顺序 =========
% theta (k), d (1), S1p (n1), S1m (n1), S2p (n2), S2m (n2),
% gamma_p (k), gamma_m (k), zeta_p (k), zeta_m (k)
nv = k + 1 + 2*n1 + 2*n2 + 2*k + 2*k;

% 索引
idx_theta   = 1:k;
idx_d       = k+1;

idx_S1p = k+1 + (1:n1);
idx_S1m = k+1 + n1 + (1:n1);
idx_S2p = k+1 + 2*n1 + (1:n2);
idx_S2m = k+1 + 2*n1 + n2 + (1:n2);

offset = k+1+2*n1+2*n2;
idx_gamma_p = offset + (1:k);
idx_gamma_m = offset + k + (1:k);
idx_zeta_p  = offset + 2*k + (1:k);
idx_zeta_m  = offset + 3*k + (1:k);

% ========= 目标函数 =========
f = zeros(nv,1);
f(idx_S1p) = 1;
f(idx_S2m) = beta;

% ========= 等式约束 =========
Aeq = [];
beq = [];

% 组1 约束: theta' * Z_j - d + S1p_j - S1m_j = 0
for t = 1:n1
    j = G1_idx(t);
    row = zeros(1,nv);
    row(idx_theta) = Zc(:,j).';
    row(idx_d)     = -1;
    row(idx_S1p(t)) = 1;
    row(idx_S1m(t)) = -1;
    Aeq = [Aeq; row];
    beq = [beq; 0];
end

% 组2 约束: theta' * Z_j - d + S2p_j - S2m_j = -eta
for t = 1:n2
    j = G2_idx(t);
    row = zeros(1,nv);
    row(idx_theta) = Zc(:,j).';
    row(idx_d)     = -1;
    row(idx_S2p(t)) = 1;
    row(idx_S2m(t)) = -1;
    Aeq = [Aeq; row];
    beq = [beq; -eta];
end

% ========= xi 加权的归一化约束 =========
%sum_i( xi_p(i)*gamma_p(i) + xi_m(i)*gamma_m(i) ) = 1
row = zeros(1,nv);
row(idx_gamma_p) = xi_p.';   % 注意转为行向量
row(idx_gamma_m) = xi_m.';
Aeq = [Aeq; row];
beq = [beq; 1];

% ========= θ 与 γ 关系: theta_i = xi_p(i)*gamma_p(i) - xi_m(i)*gamma_m(i) =========
for i = 1:k
    row = zeros(1,nv);
    row(idx_theta(i))   = 1;
    row(idx_gamma_p(i)) = -xi_p(i);
    row(idx_gamma_m(i)) =  xi_m(i);
    Aeq = [Aeq; row];
    beq = [beq; 0];
end

% ========= 不等式约束 =========
A = [];
b = [];

%  γ⁺≤ ζ⁺, γ⁻≤ ζ⁻ 
for i=1:k
    row = zeros(1,nv); row(idx_zeta_p(i))=-1; row(idx_gamma_p(i))=1;
    A = [A; row]; b = [b; 0];
    row = zeros(1,nv); row(idx_zeta_m(i))=-1; row(idx_gamma_m(i))=1;
    A = [A; row]; b = [b; 0];
end

% ζ⁺+ζ⁻ ≤ 1
for i=1:k
    row=zeros(1,nv); row(idx_zeta_p(i))=1; row(idx_zeta_m(i))=1;
    A=[A;row]; b=[b;1];
end

% sum(ζ⁺+ζ⁻) ≤ k
row=zeros(1,nv);
row(idx_zeta_p)=1; row(idx_zeta_m)=1;
A=[A;row]; b=[b;k];

% ========= 边界 =========
lb=-inf(nv,1); ub=inf(nv,1);
if isfield(opts,'vmin'); lb(idx_theta)=opts.vmin; end
lb([idx_S1p idx_S1m idx_S2p idx_S2m])=0;
lb([idx_gamma_p idx_gamma_m idx_zeta_p idx_zeta_m])=0;
ub([idx_zeta_p idx_zeta_m]) = 1;
intcon = [idx_zeta_p idx_zeta_m];  % intlinprog 的整数变量索引（必须为正整数索引向量）
% ========= 求解 =========
options = optimoptions('intlinprog','Display','none','MaxTime',300,'RelativeGapTolerance',1e-3);
[z,fval,exitflag,output] = intlinprog(f,intcon,A,b,Aeq,beq,lb,ub,options);

% ========= 提取解 =========
sol.d       = z(idx_d);
sol.S1p     = z(idx_S1p); sol.S1m = z(idx_S1m);
sol.S2p     = z(idx_S2p); sol.S2m = z(idx_S2m);
sol.gamma_p = z(idx_gamma_p); sol.gamma_m = z(idx_gamma_m);
sol.zeta_p  = z(idx_zeta_p); sol.zeta_m = z(idx_zeta_m);
sol.fval = fval; sol.exitflag = exitflag; sol.output = output;


% 强制非负（防止容差不足仍残留负值）
sol.gamma_p(sol.gamma_p < 0) = 0;
sol.gamma_m(sol.gamma_m < 0) = 0;

% 同步修正 theta：theta_i = xi_p(i)*gamma_p(i) - xi_m(i)*gamma_m(i)
sol.theta = xi_p .* sol.gamma_p - xi_m .* sol.gamma_m;

% ========= 分类判别 =========
scores = sol.theta.' * Zc;  % 1 x n
R1 = find(scores >= sol.d);
R2 = find(scores <= sol.d - eta);
R0 = setdiff(1:n, union(R1,R2));
sol.R1=R1;
sol.R2=R2;
sol.R0=R0;

d=sol.d;
% 如果存在重叠区域，调用 model8_1_8 进一步判别
if ~isempty(R0)
    C1 = intersect(R0, G1_idx); % 原来属于组1的重叠样本
    C2 = intersect(R0, G2_idx); % 原来属于组2的重叠样本
    if ~isempty(C1) || ~isempty(C2)
        % 调用 8.1.8 第二阶段模型，使用本阶段得到的 theta 作为输入
        [assignments, sol2] = model8_1_8(Z, C1, C2, d, eta, beta, sol.theta, opts);
        % 将 assignments 中被判为组1 的样本加入 R1；被判为组2 的加入 R2
        R1 = union(R1, assignments.C1_assign_idx(:).');
        R2 = union(R2, assignments.C2_assign_idx(:).');
        % 重新计算 R0
        R0 = setdiff(1:n, union(R1,R2));
        % 将子模型解存入 sol
        sol.submodel = sol2;
        sol.assignments = assignments;
    end
end

% 最终评估（F1, AUC）——仅在没有未决样本或仍想计算整体性能时有效
y_true = group(:);
y_pred = zeros(n,1);
y_pred(R1) = 0;
y_pred(R2) = 1;
% 二分类（正类 = 1）
y_true_bin = (y_true==1);
y_pred_bin = (y_pred==1);

TP = sum(y_true_bin==1 & y_pred_bin==1);
FP = sum(y_true_bin==0 & y_pred_bin==1);
FN = sum(y_true_bin==1 & y_pred_bin==0);
TN = sum(y_true_bin==0 & y_pred_bin==0); 
precision = TP / (TP+FP+eps);
recall    = TP / (TP+FN+eps);
F1 = 2 * precision * recall / (precision+recall+eps);
Accuracy = (TP + TN) / (TP + TN + FP + FN + eps);

% ========= AUC 计算：使用 DA-DEA 秩和法（Mann-Whitney U） =========
% 原理：AUC = P(score_positive > score_negative)
% 公式：AUC = (S - m(m+1)/2) / (m * n)
% 其中：S 为所有正样本排序名次之和，m 为正样本数，n 为负样本数
% 参考：Hanley & McNeil, Radiology, 1982
%
% 注意：根据分类规则，负类得分 >= d，正类得分 <= d-eta
% 因此正类得分 < 负类得分。为了正确计算AUC（正类得分应高于负类得分），
% 需要反转得分方向，使用 -scores 来计算AUC

% 使用 DA-DEA 效率得分（scores）作为连续判别得分
% scores 已经是 sol.theta.' * Zc，即每个样本的判别得分

% 确保 scores 为行向量
scores_orig = scores(:).';

% 反转得分方向用于AUC计算（使得正类得分高于负类得分时，AUC高）
% 因为分类规则是：正类得分 <= d-eta < d <= 负类得分
% 反转后：-正类得分 >= -(d-eta) > -d >= -负类得分
% 即反转后正类得分高于负类得分，符合AUC计算要求
scores_auc = -scores_orig;

% 使用平均秩处理并列（tiedrank）
% 得分越高，秩越大（升序），并列得分取平均秩
ranks = tiedrank(scores_auc);

% 找到正样本（group=1）和负样本（group=0）的索引
pos_idx = find(y_true_bin == 1);  % 正样本索引
neg_idx = find(y_true_bin == 0);  % 负样本索引

m = length(pos_idx);  % 正样本数
n_pos = length(neg_idx);  % 负样本数（避免与变量n冲突）

% 计算所有正样本的排序名次之和
S = sum(ranks(pos_idx));

% 使用秩和法计算 AUC
if m > 0 && n_pos > 0
    AUC = (S - m*(m+1)/2) / (m * n_pos);
else
    AUC = NaN;  % 如果正样本或负样本数为0，无法计算AUC
end

% 保存归一化得分（用于其他用途，使用原始得分）
score_min = min(scores_orig);
score_max = max(scores_orig);
if score_max > score_min
    score_true_norm = (scores_orig - score_min) / (score_max - score_min);
else
    score_true_norm = scores_orig;  % 如果所有得分相同
end
sol.score_norm = score_true_norm;
sol.scores = scores_orig;  % 保存原始得分

sol.F1 = F1;
sol.AUC = AUC;
sol.Accuracy = Accuracy;
sol.precision = precision;
sol.recall = recall;
end

