%% Compute Jacobian tensor from symbolic function
% f: symbolic function
% vars: symbolic variables
% X: m*N matrix of N input points
% Output:
% J_tensor: n*m*N Jacobian tensor with
% n = number of outputs
% m = number of input variables
% N = number of sample points
function J_tensor = compute_symbolic_jacobian_tensor(f, vars, X)

    m = length(vars);     
    [~, N] = size(X);     
    J_sym = jacobian(f, vars);  % symbolic Jacobian matrix

    n = size(J_sym, 1); 
    J_tensor = zeros(n, m, N);

    J_func = matlabFunction(J_sym, 'Vars', {vars(:)}); 

    for i = 1:N
        J_tensor(:, :, i) = J_func(X(:, i)); 
    end
end
