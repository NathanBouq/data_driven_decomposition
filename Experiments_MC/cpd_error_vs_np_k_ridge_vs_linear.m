%% CPD error vs N and k: ridge vs. linear regression with Monte Carlo
% f: symbolic function
% vars: symbolic variables 
% r: CP rank for CPD
% f_id: function identifier 
% k_vals: list of k nearest neighbors values 
% Ns: list of sample sizes
% num_trials: number of Monte Carlo trials
% Output:
% error_matrix: [length(Ns) x length(k_vals) x 2] CPD errors 
%               (dim 3 = 1 for linear, 2 for ridge regression)
% error_stds: same size, standard deviations across trials
function [error_matrix, error_stds] = cpd_error_vs_np_k_ridge_vs_linear(f, vars, r, f_id, k_vals, Ns, num_trials)

f_num = matlabFunction(f, 'Vars', {vars(:)});
m = length(vars);

methods = {@local_jacobian_regression, @(X, Y, k)local_jacobian_ridge(X,Y,k,1e-5)};
method_names = {'linear', 'ridge'};

error_matrix = NaN(length(Ns), length(k_vals), length(methods));
error_stds = NaN(length(Ns), length(k_vals), length(methods));

for mth = 1:length(methods)
    method = methods{mth};

    for i = 1:length(Ns)
        N = Ns(i);

        for j = 1:length(k_vals)
            k = k_vals(j);

            if k >= N
                warning("Skipping k = %d >= N = %d", k, N);
                continue;
            end

            trial_errors = zeros(1, num_trials);

            for trial = 1:num_trials
                try
                    X = sample_input_points(m, N, f_id);
                    Y = f_num(X);

                    J_true = compute_symbolic_jacobian_tensor(f, vars, X);
                    [U_ref, ~] = cpd(J_true, r);

                    J_est = method(X, Y, k);
                    [U_est, ~] = cpd(J_est, r);

                    trial_errors(trial) = norm(cpderr({U_ref{1}, U_ref{2}}, {U_est{1}, U_est{2}}));

                catch ME
                    warning("Error at N = %d, k = %d, trial %d: %s", N, k, trial, ME.message);
                    trial_errors(trial) = NaN;
                end
            end

            % avg and std deviation across trials
            error_matrix(i, j, mth) = mean(trial_errors, 'omitnan');
            error_stds(i, j, mth) = std(trial_errors, 'omitnan');
        end
    end
end

output_folder = 'Thesis_code/experiments_plots_MC';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

[min_err, idx] = min(error_matrix(:));
[i_min, j_min, m_min] = ind2sub(size(error_matrix), idx);
fprintf('🔍 Minimum CPD error = %.2e at N = %d, k = %d, method = %s\n', ...
    min_err, Ns(i_min), k_vals(j_min), method_names{m_min});

% one plot per k with both methods on same plot
for j = 1:length(k_vals)
    figure;
    hold on;
    for mth = 1:length(methods)
        errorbar(Ns, error_matrix(:, j, mth), error_stds(:, j, mth), '-o', 'LineWidth', 2, 'MarkerSize', 8, ...
            'DisplayName', method_names{mth});
    end
    set(gca, 'XScale', 'log', 'YScale', 'log');
    xlabel('Number of points (N)');
    ylabel('Avg CPD Error');
    f_str = strjoin(arrayfun(@char, f, 'UniformOutput', false), '; ');
    title(sprintf('CPD Error vs N for k = %d\nfor f = [%s]', k_vals(j), f_str));
    legend('Location', 'northeast');
    grid on;

    filename = sprintf('cpd_error_vs_n_combined_k%d_fid_%d.png', k_vals(j), f_id);
    saveas(gcf, fullfile(output_folder, filename));
end
end
