%% Local Jacobian estimation via polynomial regression
% X: m*N matrix of N input points
% Y: n*N matrix of corresponding outputs
% k: number of nearest neighbors
% degree: degree of local polynomial (e.g., 1 = linear, 2 = quadratic)
% Output:
% J_tensor: n*m*N Jacobian tensor with
% n = number of outputs
% m = number of input variables
% N = number of sample points
function J_tensor = local_poly_regression(X, Y, k, degree)

    [m, N] = size(X);
    n = size(Y, 1);
    J_tensor = NaN(n, m, N);
    
    multi_idx = generate_monomial_indices(m, degree);
    num_features = size(multi_idx, 1);
    
    for i = 1:N
        xi = X(:, i);
    
        dists = vecnorm(X - xi, 2, 1); % k nearest neighbors excluding self
        [~, neighbors] = mink(dists, k + 1); 
    
        % compute differences
        DX = X(:, neighbors) - xi;  
        DY = Y(:, neighbors) - Y(:, i); 
    
        Phi = zeros(k + 1, num_features);
        for j = 1:(k + 1)
            Phi(j, :) = compute_polynomial_features(DX(:, j), multi_idx);
        end
        
        % solve linear regression for each output
        for l = 1:n
            y_vec = DY(l, :)';  % (k+1) × 1
            coeffs = pinv(Phi) * y_vec;
    
            % Extract Jacobian row from monomials
            J_row = zeros(1, m);
            unit_vec = eye(m);
            for dim = 1:m
                linear_idx = find(ismember(multi_idx, unit_vec(dim, :), 'rows'));
                if ~isempty(linear_idx)
                    J_row(dim) = coeffs(linear_idx);
                end
            end
            J_tensor(l, :, i) = J_row;
        end
    end
end

%% Generate monomial multi-indices up to given degree
% m: number of input variables
% d: maximum total degree of monomials
% Output:
% M: K*m matrix of multi-indices (K monomials, m variables)
function M = generate_monomial_indices(m, d)

    M = [];
    for total_deg = 1:d  % skip constant term since Jacobian uses derivatives
        M = [M; generate_degree_combinations(m, total_deg)];
    end
end

%% Generate exponent combinations summing to fixed degree
% m: number of input variables
% d: total degree
% Output:
% combs: K*m matrix of non-negative integer vectors summing to d
function combs = generate_degree_combinations(m, d)

combs = []; 
    if d == 0
        combs = zeros(1, m);
    else
        if m == 1
            combs = d;
        else
            for i = 0:d
                tail = generate_degree_combinations(m - 1, d - i);
                combs = [combs; [i * ones(size(tail, 1), 1), tail]];
            end
        end
    end
end

%% Compute monomial features from input and exponent vectors
% x: m*1 input vector
% multi_idx: K*m matrix of exponent vectors (multi-indices)
% Output:
% phi: 1*K row vector of monomial terms evaluated at x
function phi = compute_polynomial_features(x, multi_idx)

    K = size(multi_idx, 1);
    phi = zeros(1, K);
    for i = 1:K
        phi(i) = prod(x'.^multi_idx(i, :));
    end
end
