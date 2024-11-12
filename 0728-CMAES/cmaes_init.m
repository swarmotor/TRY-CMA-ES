function [x0, sigma, D, C] = cmaes_init(nVar, lb, ub, options)
    % 这个函数初始化CMA-ES算法的参数

    % 设置变量数量
    nVar = nVar;

    % 设置变量界限
    lb = lb;
    ub = ub;

    % 从选项中获取最大函数评估次数
    MaxFunEvals = options.MaxFunEvals;

    % 随机初始化起始点
    x0 = lb + (ub - lb) * rand(1, nVar);

    % 初始化步长
    sigma = (ub - lb) / 6;

    % 初始化协方差矩阵为单位矩阵
    C = eye(nVar);

    % 初始化D矩阵，这里简化为C的平方根
    D = sqrt(sigma) * C;
end