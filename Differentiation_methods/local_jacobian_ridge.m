%% Local Jacobian estimation via ridge regression
% X: m*N matrix of N input points
% Y: n*N matrix of corresponding outputs
% k: number of nearest neighbors
% lambda: regularization parameter (e.g., 1e-3)
% Output:
% J_tensor: n*m*N Jacobian tensor with
% n = number of outputs
% m = number of input variables
% N = number of sample points
function J_tensor = local_jacobian_ridge(X, Y, k, lambda)

    [m, N] = size(X);
    [n, ~] = size(Y);
    J_tensor = zeros(n, m, N);

    D = squareform(pdist(X')); % pairwise distances between points

    for i = 1:N
        D(i, i) = inf;  % exclude self
        [~, idx] = sort(D(i, :), 'ascend');
        neighbors = idx(1:k);  % indices of k nearest neighbors

        % compute differences
        dX = X(:, neighbors) - X(:, i);   
        dY = Y(:, neighbors) - Y(:, i); 

        G = dX * dX';      
        I = eye(m);
        J_i = dY * dX' / (G + lambda * I); 
        J_tensor(:, :, i) = J_i;
    end
end
