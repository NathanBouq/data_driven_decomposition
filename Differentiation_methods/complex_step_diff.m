%% Finite difference: complex step difference method
% f: numerical function
% X: m*N matrix of N sample points and m input variables 
% h: step size 
% Ouput:
% J_tensor: n*m*N Jacobian tensor with 
% n = number of outputs of f
% m = number of input variables
% N = number of sample points
function J_tensor = complex_step_diff(f, X, h)

    [m, N] = size(X);
    test_output = f(X(:, 1));
    n = length(test_output);
    J_tensor = zeros(n, m, N);
    
    for k = 1:N
        x0 = X(:, k);
        for i = 1:m
            x_step = x0;
            x_step(i) = x_step(i) + 1i * h;  
            f_step = f(x_step);              
            J_tensor(:, i, k) = imag(f_step) / h;  
        end
    end
end
