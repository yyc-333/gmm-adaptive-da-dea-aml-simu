function [assignments, sol2] = model8_1_8(Z, C1, C2, d, eta, beta, theta_input, opts)
% MODEL8_1_8  第二阶段：重叠样本再优化（带 θ 微调正则项）
%
% 优化目标：
%   min  sum(S1p) + beta*sum(S2m) + lambda*(alpha*||theta-theta1||_1+(1-alpha)*||theta-theta1||_2^2)
%
% 其中 theta1 为第一阶段得到的解

%% ========= 准备 =========
if nargin < 8
    opts = struct();
end

lambda = 0.1;   % 正则强度（建议 0.1 ~ 5 之间调参）
alpha_en = 0.8;   % Elastic Net L1比例
theta1 = theta_input(:);

[n,k] = size(Z);
Zc = Z.';   % k × n

C1 = C1(:).';
C2 = C2(:).';

nC1 = numel(C1);
nC2 = numel(C2);

%% ========= 变量索引 =========
% z = [S1p S1m S2p S2m c theta]

idx_S1p = 1:nC1;
idx_S1m = nC1 + (1:nC1);
idx_S2p = 2*nC1 + (1:nC2);
idx_S2m = 2*nC1 + nC2 + (1:nC2);
idx_c   = 2*nC1 + 2*nC2 + 1;
idx_theta = idx_c + (1:k);

% Elastic Net auxiliary variable
idx_u = idx_theta(end) + (1:k);

nv = idx_u(end);

%% ========= 构造目标函数 =========
H = zeros(nv);
f = zeros(nv,1);

% 线性部分
if nC1 > 0
    f(idx_S1p) = 1;
end
if nC2 > 0
    f(idx_S2m) = beta;
end

% ===== 二次正则项 λ||θ-θ1||^2 =====
% quadprog 形式: 1/2 z'H z + f'z
% 所以 H 放 2λI

lambda1 = lambda*alpha_en;
lambda2 = lambda*(1-alpha_en);

% Elastic Net L2 component
H(idx_theta, idx_theta) = 2*lambda2*eye(k);
f(idx_theta) = -2*lambda2*theta1;

% Elastic Net L1 component
f(idx_u) = lambda1;

%% ========= 等式约束 =========
Aeq = [];
beq = [];

% C1 样本
for ii = 1:nC1
    j = C1(ii);
    row = zeros(1,nv);
    row(idx_S1p(ii)) = 1;
    row(idx_S1m(ii)) = -1;
    row(idx_c) = -1;
    row(idx_theta) = Zc(:,j)';
    Aeq = [Aeq; row];
    beq = [beq; 0];
end

% C2 样本
for ii = 1:nC2
    j = C2(ii);
    row = zeros(1,nv);
    row(idx_S2p(ii)) = 1;
    row(idx_S2m(ii)) = -1;
    row(idx_c) = -1;
    row(idx_theta) = Zc(:,j)';
    Aeq = [Aeq; row];
    beq = [beq; 0];
end

%% ========= 不等式约束 =========
A = [];
b = [];

% Elastic Net L1 constraints
% u >= |theta-theta1|

for j = 1:k

    row = zeros(1,nv);
    row(idx_theta(j)) = 1;
    row(idx_u(j)) = -1;
    A = [A; row];
    b = [b; theta1(j)];

    row = zeros(1,nv);
    row(idx_theta(j)) = -1;
    row(idx_u(j)) = -1;
    A = [A; row];
    b = [b; -theta1(j)];

end

% c ≤ d
row = zeros(1,nv);
row(idx_c) = 1;
A = [A; row];
b = [b; d];

% c ≥ d-eta  →  -c ≤ -(d-eta)
row = zeros(1,nv);
row(idx_c) = -1;
A = [A; row];
b = [b; -(d-eta)];

%% ========= 变量上下界 =========
lb = -inf(nv,1);
ub =  inf(nv,1);

if nC1 > 0
    lb(idx_S1p) = 0;
    lb(idx_S1m) = 0;
end
if nC2 > 0
    lb(idx_S2p) = 0;
    lb(idx_S2m) = 0;
end

lb(idx_c) = d - eta;
ub(idx_c) = d;

% Elastic Net auxiliary variable lower bound
lb(idx_u) = 0;

%% ========= 求解 =========
options = optimoptions('quadprog', ...
    'Algorithm','interior-point-convex', ...
    'Display','none', ...
    'OptimalityTolerance',1e-8);

[z,fval,exitflag,output] = quadprog(H,f,A,b,Aeq,beq,lb,ub,[],options);

if exitflag <= 0
    warning('quadprog did not converge. exitflag=%d',exitflag);
end

%% ========= 提取结果 =========
theta_new = z(idx_theta);
c_scalar  = z(idx_c);

sol2.theta = theta_new;
sol2.S1p   = z(idx_S1p);
sol2.S1m   = z(idx_S1m);
sol2.S2p   = z(idx_S2p);
sol2.S2m   = z(idx_S2m);
sol2.c     = c_scalar;
sol2.fval  = fval;
sol2.exitflag = exitflag;
sol2.output = output;
sol2.theta_deviation_L1 = norm(theta_new-theta1,1);
sol2.theta_deviation_L2 = norm(theta_new-theta1,2);

%% ========= 分配规则 =========
scores_C = theta_new.' * Zc(:,[C1 C2]);
scores_C = scores_C(:);

C1_assign_mask = (scores_C > c_scalar);
C2_assign_mask = (scores_C <= c_scalar);

assignments.C1_assign_idx = ...
    [C1(C1_assign_mask(1:nC1)), ...
     C2(C2_assign_mask(nC1+1:end))];

assignments.C2_assign_idx = ...
    [C1(C2_assign_mask(1:nC1)), ...
     C2(C1_assign_mask(nC1+1:end))];

assignments.C1_assign_idx = assignments.C1_assign_idx(:)';
assignments.C2_assign_idx = assignments.C2_assign_idx(:)';

assignments.c = c_scalar;
assignments.d = d;
assignments.eta = eta;

end