% CMA-ES参数设置
lb = 1; % h的下界
ub = 3; % h的上界
options.MaxFunEvals = 500; % 最大函数评估次数

% 确保lambda是在合适的地方定义的
lambda = 4 + round(3 * log(1)); % 例如，使用CMA-ES的标准公式

% 初始化CMA-ES的变量
x0 = lb + (ub - lb) * rand(1, 1); % 假设h是在[lb, ub]范围内的随机起始点
sigma = (ub - lb) / 6; % 初始步长
D = sqrt(sigma) * eye(1); % 初始化D矩阵，这里假设只有一个参数，即h
C = eye(1); % 初始化协方差矩阵


% CMA-ES主循环
for generation = 1:options.MaxFunEvals
    fitness_values = zeros(lambda, 1);
    [sigma, mu, lambda, pc, ps, C] = CMAES(options, @my_fitness_function);
    disp(generation);
    disp(fitness_values);
end

% 找到最佳适应度和对应的h值
[best_fitness, best_idx] = min(fitness_values);z
best_para = x0 ;

% 输出最佳h值
fprintf('找到的最佳值为: %f, 对应的适应度为: %f\n', best_para, -best_fitness);