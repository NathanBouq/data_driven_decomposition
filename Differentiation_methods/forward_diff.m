%% Finite difference: forward difference method
% f: numerical function
% X: m*N matrix of N sample points and m input variables 
% h: step size 
% Ouput:
% J_tensor: n*m*N Jacobian tensor with 
% n = number of outputs of f
% m = number of input variables
% N = number of sample points
function J_tensor = forward_diff(f, X, h)

    [m, N] = size(X);           
    test = f(X(:,1));    
    n = length(test);    
    J_tensor = zeros(n, m, N);  

    for k = 1:N
        x0 = X(:, k);           
        f0 = f(x0);             

        for i = 1:m
            x_perturbed = x0;
            x_perturbed(i) = x_perturbed(i) + h;   
            f_perturbed = f(x_perturbed);           
            J_tensor(:, i, k) = (f_perturbed - f0) / h;  
        end
    end
end

