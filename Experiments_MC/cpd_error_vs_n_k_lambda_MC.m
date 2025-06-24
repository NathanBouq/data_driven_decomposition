%% CPD error over N, k, lambda using ridge regression with Monte Carlo
% f: symbolic function
% vars: symbolic variables 
% r: CP rank for CPD
% f_id: identifier for function 
% Ns: list of sample sizes to test
% k_vals: list of k nearest neighbors values 
% lambdas: list of regularization parameters
% num_trials: number of Monte Carlo trials
% Output:
% error_tensor: [length(Ns) x length(k_vals) x length(lambdas)] CPD errors
% error_stds: same size, standard deviations across trials
function [error_tensor, error_stds] = cpd_error_vs_n_k_lambda_MC(f, vars, r, f_id, Ns, k_vals, lambdas, num_trials)

f_num = matlabFunction(f, 'Vars', {vars(:)});
m = length(vars);

error_tensor = NaN(length(Ns), length(k_vals), length(lambdas));
error_stds = NaN(length(Ns), length(k_vals), length(lambdas));

for i = 1:length(Ns)
    N = Ns(i);

    for j = 1:length(k_vals)
        k = k_vals(j);
        if k >= N
            continue;
        end

        for l = 1:length(lambdas)
            lambda = lambdas(l);
            trial_errors = zeros(1, num_trials);

            for trial = 1:num_trials
                try
                    X = sample_input_points(m, N, f_id);
                    Y = f_num(X);

                    J_true = compute_symbolic_jacobian_tensor(f, vars, X);
                    [U_ref, ~] = cpd(J_true, r);

                    J_est = local_jacobian_ridge(X, Y, k, lambda);
                    [U_est, ~] = cpd(J_est, r);

                    trial_errors(trial) = norm(cpderr({U_ref{1}, U_ref{2}}, {U_est{1}, U_est{2}}));

                catch ME
                    warning("Error at N = %d, k = %d, lambda = %.1e (trial %d): %s", ...
                        N, k, lambda, trial, ME.message);
                    trial_errors(trial) = NaN;
                end
            end

            % avg and std deviation across trials
            error_tensor(i, j, l) = mean(trial_errors, 'omitnan');
            error_stds(i, j, l) = std(trial_errors, 'omitnan');
        end
    end
end

output_folder = 'Thesis_code/experiments_plots_MC';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

[min_err, idx] = min(error_tensor(:));
[i_min, j_min, l_min] = ind2sub(size(error_tensor), idx);
fprintf('🔍 Minimum CPD error = %.2e at N = %d, k = %d, lambda = %.1e\n', ...
    min_err, Ns(i_min), k_vals(j_min), lambdas(l_min));

% one plot per lambda
for l = 1:length(lambdas)
    figure;
    hold on;
    for j = 1:length(k_vals)
        % with error bars
        errorbar(Ns, squeeze(error_tensor(:, j, l)), squeeze(error_stds(:, j, l)), '-o', ...
            'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', sprintf('k = %d', k_vals(j)));
    end
    xlabel('Number of points (N)');
    ylabel('Avg CPD Error');
    f_str = strjoin(arrayfun(@char, f, 'UniformOutput', false), '; ');
    title(sprintf('CPD Error vs N (λ = %.1e) with Monte Carlo for f = [%s]', lambdas(l), ...
        f_str));
    set(gca, 'XScale', 'log', 'YScale', 'log');
    legend('Location', 'northeast');
    grid on;
    hold off;

    filename = sprintf('cpd_error_vs_N_lambda_%s_f_id_%d.png', lambdas(l), f_id);
    saveas(gcf, fullfile(output_folder, filename));
end

end
