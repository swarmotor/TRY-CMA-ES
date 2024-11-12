function [sigma, mu, lambda, pc, ps, C] = CMAES(options, CostFunction)
clc;
clear;
close all;
CostFunction = @my_fitness_function; % 目标函数，这里使用的是球面函数
% 问题维度
nVar = 1;                % 决策变量的数量
VarSize = [1 nVar];       % 决策变量矩阵的大小

% 变量的界限
VarMin = 0  ;          
VarMax = 10;         
% 初始化参数
MaxIt = 200; % 最大迭代次数
lambda = 10; % 后代数量
mu = round(lambda / 2); % 父代数量

% 父代权重
w = log(mu + 0.5) - log(1:mu); % 权重计算
w = w / sum(w);               % 权重归一化

% 有效父代数量
mu_eff = 1 / sum(w.^2);

% 步长控制参数
sigma0 = 0.5 * (VarMax - VarMin); % 初始步长
cs = (mu_eff + 2) / (nVar + mu_eff + 5); % 步长衰减参数
ds = 1 + cs + 2 * max(sqrt((mu_eff - 1) / (nVar + 1)) - 1, 0); % 步长调整参数
ENN = sqrt(nVar) * (1 - 1 / (4 * nVar) + 1 / (21 * nVar^2)); % 期望的归一化步长

% 协方差矩阵更新参数
cc = (4 + mu_eff / nVar) / (4 + nVar + 2 * mu_eff / nVar); % 协方差矩阵更新的学习率
c1 = 2 / ((nVar + 1.3)^2 + mu_eff); % 路径样本的权重
alpha_mu = 2; % 一个用于控制cmu的常数
cmu = min(1 - c1, alpha_mu * (mu_eff - 2 + 1 / mu_eff) / ((nVar + 2)^2 + alpha_mu * mu_eff / 2)); % 权重更新参数
hth = (1.4 + 2 / (nVar + 1)) * ENN; % 用于调整协方差矩阵的阈值

% 种群初始化
ps = cell(MaxIt, 1); % 进化路径向量初始化
pc = cell(MaxIt, 1); % 权重进化路径向量初始化
C = cell(MaxIt, 1); % 协方差矩阵初始化
sigma = cell(MaxIt, 1); % 步长初始化

% 初始化种群
ps{1} = zeros(VarSize); % 第一代的进化路径向量初始化为0
pc{1} = zeros(VarSize); % 第一代的权重进化路径向量初始化为0
C{1} = eye(nVar); % 第一代的协方差矩阵初始化为单位矩阵
sigma{1} = sigma0; % 第一代的步长初始化

% 定义空个体结构
empty_individual.Position = [];
empty_individual.Step = [];
empty_individual.Cost = [];

% 种群初始化
M = repmat(empty_individual, MaxIt, 1); % 创建种群数组
M(1).Position = unifrnd(VarMin, VarMax, VarSize); % 第一个个体的位置随机初始化
M(1).Step = zeros(VarSize); % 第一个个体的步长初始化为0
M(1).Cost = CostFunction(M(1).Position); % 计算第一个个体的成本

% 初始化最佳解和最佳成本
BestSol = M(1); % 当前最佳解初始化为第一个个体
BestCost = zeros(MaxIt, 1); % 最佳成本初始化为0数组
%% 主循环
for g = 1:MaxIt
    %生成样本
    pop = repmat(empty_individual, lambda, 1);

    for i = 1:lambda
        pop(i).Step = mvnrnd(zeros(VarSize), C{g});
        pop(i).Position = M(g).Position + sigma{g}*pop(i).Step;
        pop(i).Position = max(pop(i).Position, VarMin);
        pop(i).Position = min(pop(i).Position, VarMax);
        pop(i).Cost = CostFunction(pop(i).Position);
        if pop(i).Cost < BestSol.Cost
            BestSol = pop(i);
        end
    end
    
    
    Costs = [pop.Cost];
    [Costs, SortOrder] = sort(Costs);
    pop = pop(SortOrder);
    BestCost(g) = BestSol.Cost;
 
    M(g+1).Step = 0;
    for j = 1:mu
        M(g+1).Step = M(g+1).Step + w(j)*pop(j).Step;
    end
    
    M(g+1).Position = M(g).Position + sigma{g}*M(g+1).Step;
    M(g+1).Position = max(M(g+1).Position, VarMin);
    M(g+1).Position = min(M(g+1).Position, VarMax);
    M(g+1).Cost = CostFunction(M(g+1).Position);
    if M(g+1).Cost < BestSol.Cost
        BestSol = M(g+1);
    end
    ps{g+1} = (1-cs)*ps{g} + sqrt(cs*(2-cs)*mu_eff)*M(g+1).Step/chol(C{g})';
    sigma{g+1} = sigma{g}*exp(cs/ds*(norm(ps{g+1})/ENN-1))^0.3;
    % (更新协方差矩阵
    if norm(ps{g+1})/sqrt(1-(1-cs)^(2*(g+1))) < hth%在不同阶段根据当前的搜索情况动态调整步长和协方差矩阵的更新方式
        hs = 1;
    else
        hs = 0;
    end

    delta = (1-hs)*cc*(2-cc);
    pc{g+1} = (1-cc)*pc{g} + hs*sqrt(cc*(2-cc)*mu_eff)*M(g+1).Step;
    C{g+1} = (1-c1-cmu)*C{g} + c1*(pc{g+1}'*pc{g+1} + delta*C{g});
    %使得算法能够动态地调整其搜索方向和形状，以适应目标函数的地形。
    for j = 1:mu
        C{g+1} = C{g+1} + cmu*w(j)*pop(j).Step'*pop(j).Step;
    end
    
    % (如果协方差矩阵不是正定的或接近奇异)
    [V, E] = eig(C{g+1});
    if any(diag(E) < 0)
        E = max(E, 0);
        C{g+1} = V*E/V;
    end
end


figure;
semilogy(BestCost, 'LineWidth', 2);
xlabel('Iteration');
ylabel('Best Cost');
grid on;
end