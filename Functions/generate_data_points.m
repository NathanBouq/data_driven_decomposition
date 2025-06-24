%% Generate input-output data from symbolic vector function
% F: symbolic function
% vars: symbolic variables
% N: number of random input points to generate
% Output:
% X: m*N matrix of input points sampled from [0, 1]^m
% Y: n*N matrix of corresponding function outputs
function [X, Y] = generate_data_points(F, vars, N)

    m = length(vars);    
    f_num = matlabFunction(F, 'Vars', {vars(:)});  

    X = rand(m, N);
    Y = f_num(X);
end
