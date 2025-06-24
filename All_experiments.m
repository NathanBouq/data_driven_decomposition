%% Script to Run All Experiments
clear all; close all; clc

addpath('Experiments', 'Functions', 'Differentiation_methods', 'Experiments_MC');
addpath(genpath(fullfile(getenv('MATLABDRIVE'), 'tensorlab')));

output_folder = 'Thesis_code/experiments_plots';
if exist(output_folder, 'dir')
    %delete(fullfile(output_folder, '*'));
end


% Define function and variables
syms x y z; 
f_id = 1; % function id (1 = polynomial, 2,3 = complex non linear functions)
[f, vars, r, degree] = get_function(f_id);
N = 50; % number of sample points
X = sample_input_points(length(vars), N, f_id); 
Ns = [50, 100, 200, 400, 800, 1600, 2400, 3000]; % list of different sample points
k_vals = [5, 10, 20, 30, 50]; % list of different k values
lambdas = logspace(-5, -1, 5); % list of different lambdas for ridge regression
num_trials = 20;  % number of Monte Carlo repetitions

%% Finite difference methods used
methods = {@forward_diff, @backward_diff, @central_diff, @complex_step_diff};
method_names = {'Forward', 'Backward', 'Central', 'Complex'};

% Visualize the function at sample points
% visualize_function_and_points(f, X, vars);

% Run experiment: CPD Error vs Step Size
% errors = cpd_error_vs_step_size_MC(f, vars, X, r, f_id, methods, method_names, num_trials);

% Run experiments: jacobian slice errors
% errors = jacobian_slice_errors_MC(f, vars, X, r, f_id, methods, method_names, num_trials); 

% Run experiment: CPD Error vs Number of Points
% cpd_error_vs_npoints_MC(f, vars, r, f_id, methods, method_names, num_trials);

% Run experiment: Factor Matrix Evolution vs Step Size
% factor_matrix_vs_step_size_MC(f, vars, X, r, f_id, methods, method_names, num_trials);

% Run experiments: cpd error vs number of points vs k nearest neighbors
% [error_matrix, error_stds] = cpd_error_vs_npoints_and_kneighbors_MC(f, vars, r, f_id, k_vals, Ns, num_trials);

% Run experiments: cpd error vs number of points vs k nearest neighbors but
% one plot for all methods and one k
% [error_matrix, error_stds] = cpd_error_vs_n_k_MC_all(f, vars, r, f_id, k_vals, Ns, num_trials);

% Run experiments: cpd error vs Ns, K neighbors, lambda
% cpd_error_vs_n_k_lambda_MC(f, vars, r, f_id, Ns, k_vals, lambdas, num_trials);

% Run experiments: cpd error vs nb of points and k nearest neighbors one plot with ridge and local linear regression
% errors = cpd_error_vs_np_k_ridge_vs_linear(f, vars, r, f_id, k_vals, Ns, num_trials);
% min_local = min(errors(:, :, 1), [], 'all')
% min_local2 = min(errors(:, :, 2), [], 'all')

%% Experiments for global polynomial fit

% Run experiment: CPD Error vs Step Size after fitting the function
% errors = cpd_error_vs_step_size_poly_MC(f, vars, X, r, f_id, num_trials, degree);

% Run experiments: jacobian slice errors after fitting the function
% jacobian_slice_errors_poly_MC(f, vars, X, r, f_id, num_trials, degree); 

% Run experiment: CPD Error vs Number of Points after fitting the function
% cpd_error_vs_npoints_poly_MC(f, vars, r, f_id, num_trials, degree);

% Run experiment: Factor Matrix Evolution vs Step Size after fitting the function
% factor_matrix_vs_step_size_poly_MC(f, vars, X, r, f_id, num_trials, degree);

f_num = matlabFunction(f, 'Vars', {vars(:)});
Y = f_num(X);
[f_hat, J_ten] = fit_polynomial_model(X, Y, degree);
method_name = 'Complex with PolyFit';
% f_approx_vals = f_hat(X);
% Run experiment: Visualize decoupling
% visualize_decoupling(f_num, X, f_approx_vals, f, method_name);

tensor = compute_symbolic_jacobian_tensor(f, vars, X);
[U, ~] = cpd(tensor, r);
W_est = U{1}
V_est = U{2}
H_est = U{3};

Y = double(subs(f, vars, num2cell(X, 1)));
g = recover_g_from_vandermonde(W_est, V_est, X, Y, d);