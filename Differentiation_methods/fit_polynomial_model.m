%% Polynomial regression fit (Vandermonde + least squares)
% X: m*N matrix of N sample points (inputs)
% Y: n*N matrix of N sample outputs
% degree: degree of the polynomial fit
% Output:
% f_hat: function handle 
% J_tensor: n*m*N Jacobian tensor with
% n = number of outputs
% m = number of input variables
% N = number of sample points
function [f_hat, J_tensor] = fit_polynomial_model(X, Y, degree)
    [m, N] = size(X);         
    [n, ~] = size(Y);        
    powers = generate_powers(m, degree);
    P = size(powers, 1); % number of monomials

    % Build Vandermonde matrix
    V = zeros(N, P);
    for i = 1:P
        term = ones(1, N);
        for j = 1:m
            term = term .* X(j, :) .^ powers(i, j);
        end
        V(:, i) = term';
    end

    coeffs = V \ Y';           % size: P x n
    coeffs = coeffs';          % size: n x P

    f_hat = @(x) evaluate_polynomial_model(x, coeffs, powers);

    J_tensor = zeros(n, m, N);
    for sample_idx = 1:N
        x_sample = X(:, sample_idx);

        for var_idx = 1:m
            dphi = zeros(1, P);
            for p = 1:P
                if powers(p, var_idx) == 0
                    dphi(p) = 0;
                else
                    new_exp = powers(p, var_idx) - 1;
                    partial_term = powers(p, var_idx);
                    for j = 1:m
                        if j == var_idx
                            partial_term = partial_term * x_sample(j)^new_exp;
                        else
                            partial_term = partial_term * x_sample(j)^powers(p, j);
                        end
                    end
                    dphi(p) = partial_term;
                end
            end
            % Jacobian slice for variable var_idx
            J_tensor(:, var_idx, sample_idx) = coeffs * dphi';
        end
    end
end

%% Generate monomial powers up to a given degree
% m: number of input variables
% degree: maximum degree of monomials
% Output:
% powers: P*d matrix of integer exponents
function powers = generate_powers(m, degree)
    powers = []; 
    for d = 0:degree
        pow = generate_fixed_degree_powers(m, d);
        powers = [powers; pow];
    end
end

%% Generate monomial powers with fixed total degree (recursive)
% m: number of input variables
% d: total degree of the monomials
% Output:
% powers: P*m matrix of exponent vectors adding up to d
function powers = generate_fixed_degree_powers(m, d)
    if m == 1
        powers = d;
    else
        powers = [];
        for i = 0:d
            back = generate_fixed_degree_powers(m-1, d-i);
            powers = [powers; [i * ones(size(back, 1), 1), back]];
        end
    end
end

%% Evaluate polynomial model at given input points
% x: m*K matrix of K input points (m variables)
% coeffs: n*P matrix of polynomial coefficients
% powers: P*m matrix of exponent vectors
% Output:
% Y: n*K matrix of model outputs
function Y = evaluate_polynomial_model(x, coeffs, powers)
    
    [m, K] = size(x);
    P = size(powers, 1);

    Vx = zeros(K, P);
    for i = 1:P
        term = ones(1, K);
        for j = 1:m
            term = term .* x(j, :) .^ powers(i, j);
        end
        Vx(:, i) = term';
    end

    Y = (Vx * coeffs)';  % n x K
end


