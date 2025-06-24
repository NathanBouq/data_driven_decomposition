%% Local Jacobian estimation via linear regression
% X: m*N matrix of N sample points (inputs)
% Y: n*N matrix of N sample outputs
% k: number of nearest neighbors for local fit
% Output:
% J_tensor: n*m*N Jacobian tensor with
% n = number of outputs
% m = number of input variables
% N = number of sample points
function J_tensor = local_jacobian_regression(X, Y, k)

    [m, N] = size(X);
    [n, ~] = size(Y);
    J_tensor = zeros(n, m, N);

    D = squareform(pdist(X')); % pairwise distances between points

    for i = 1:N
        D(i, i) = inf;  % exclude self because it will be 0
        [~, idx] = sort(D(i, :), 'ascend');
        neighbors = idx(1:k);  % indices of k nearest neighbors

        % compute differences
        dX = X(:, neighbors) - X(:, i);   % m x k
        dY = Y(:, neighbors) - Y(:, i);   % n x k

        J_i = dY / dX; 
        J_tensor(:, :, i) = J_i;
    end
end
