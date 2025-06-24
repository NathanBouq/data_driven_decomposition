%% Script to test the symbolic pipeline

syms x y;
[f, vars, r] = get_function(1);
N = 50; % number of sample points
X = rand(length(vars), N); % sample input points

% build the symbolic jacobian tensor
J_sym = compute_symbolic_jacobian_tensor(f, vars, X);
% Perform the CPD
[U, ~] = cpd(J_sym, r);

% reconstruct the tensor from the CPD
J_recon = cpdgen(U);

% compare the reconstructed tensor with the symbolic tensor
cpd_error = norm(J_sym(:) - J_recon(:)) / norm(J_sym(:));
disp(['Cpd error = ', num2str(cpd_error)])

% Final check: confirm error is near machine precision
if norm(cpd_error) < 1e-10
    disp('Symbolic CPD works correctly (up to machine precision).');
else
    disp('Warning: symbolic CPD error too large — check implementation.');
end

% compare with the ground truth
W_true = eye(2);
V_true = [1 1; 1 -1];
h1 = sum(X, 1);        % h1(x) = x + y
h2 = X(1, :) - X(2, :); % h2(x) = x - y
H_true = [h1'.^2, h2'.^3];  % N × r
[alignment_error, perm, scales] = cpderr({W_true, V_true, H_true}, {U{1}, U{2}, U{3}});
disp(['Factor alignment error = ', num2str(alignment_error)])

%% Part 2: recovering the original decoupled sctructure W, V, g

if size(perm, 1) > 1  % if it's a matrix
    [~, perm] = max(perm, [], 1);
end

W_recon = U{1}(:, perm);
V_recon = U{2}(:, perm);
H_recon = U{3}(:, perm);

% Estimate component-wise norms from the columns
scaling_est = vecnorm(W_recon) .* vecnorm(V_recon); 

% apply scaling
for i = 1:r
    W_recon(:, i) = W_recon(:, i) / norm(W_recon(:, i));
    V_recon(:, i) = V_recon(:, i) / norm(V_recon(:, i));
    H_recon(:, i) = H_recon(:, i) * scaling_est(i);
end

