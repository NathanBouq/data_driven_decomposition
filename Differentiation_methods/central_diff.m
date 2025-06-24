%% Finite difference: central difference method
% f: numerical function
% X: m*N matrix of N sample points and m input variables 
% h: step size 
% Ouput:
% J_tensor: n*m*N Jacobian tensor with 
% n = number of outputs of f
% m = number of input variables
% N = number of sample points
function J_tensor = central_diff(f, X, h)

    [m, N] = size(X);
    test = f(X(:,1));
    n = length(test);
    J_tensor = zeros(n, m, N);

    for k = 1:N
        x0 = X(:, k);
        
        for i = 1:m
            x_plus = x0;
            x_minus = x0;
            x_plus(i) = x_plus(i) + h;
            x_minus(i) = x_minus(i) - h;

            f_plus = f(x_plus);
            f_minus = f(x_minus);
            J_tensor(:, i, k) = (f_plus - f_minus) / (2 * h);
        end
    end
end
