%% CPD factor matrix error vs step size using polynomial regression with Monte Carlo
% f: symbolic function
% vars: symbolic variables
% X_init: m*N matrix of initial input points
% r: CP rank for CPD
% f_id: function identifier 
% num_trials: number of Monte Carlo trials
% degree: degree of the polynomial regression
function factor_matrix_vs_step_size_poly_MC(f, vars, X_init, r, f_id, num_trials, degree)

step_sizes = logspace(-5, -2, 4);
m = length(vars);
N = size(X_init, 2);

f_num = matlabFunction(f, 'Vars', {vars(:)});

V_error = zeros(length(step_sizes), 1);
W_error = zeros(length(step_sizes), 1);
V_error_stds = zeros(length(step_sizes), 1);
W_error_stds = zeros(length(step_sizes), 1);

for i = 1:length(step_sizes)
    h = step_sizes(i);
    trial_V_errors = zeros(1, num_trials);
    trial_W_errors = zeros(1, num_trials);

    for trial = 1:num_trials
        try
            X = sample_input_points(m, N, f_id);
            Y = f_num(X);
            [f_poly, J_est] = fit_polynomial_model(X, Y, degree);

            Jo_tensor = compute_symbolic_jacobian_tensor(f, vars, X);
            [Uo, ~] = cpd(Jo_tensor, r);
            Wo = Uo{1}; Vo = Uo{2};

            [U_est, ~] = cpd(J_est, r);
            W_est = U_est{1}; V_est = U_est{2};

            [~, perm, scales] = cpderr({Wo, Vo}, {W_est, V_est});
            if isempty(perm)
                warning('cpderr failed to align factors at h=%.1e', h);
                trial_V_errors(trial) = NaN;
                trial_W_errors(trial) = NaN;
                continue;
            end
            if size(perm, 1) > 1
                [~, perm] = max(perm, [], 1);
            end
            % order and scale
            W_est = W_est(:, perm);
            V_est = V_est(:, perm);
            for k = 1:r
                scale_w = (Wo(:, k)' * W_est(:, k)) / norm(W_est(:, k))^2;
                scale_v = (Vo(:, k)' * V_est(:, k)) / norm(V_est(:, k))^2;
                W_est(:, k) = W_est(:, k) * scale_w;
                V_est(:, k) = V_est(:, k) * scale_v;
            end

            trial_W_errors(trial) = norm(Wo - W_est, 'fro') / norm(Wo, 'fro');
            trial_V_errors(trial) = norm(Vo - V_est, 'fro') / norm(Vo, 'fro');

        catch ME
            warning("Failed at h = %.1e using %s (trial %d): %s", h, method_name, trial, ME.message);
            trial_W_errors(trial) = NaN;
            trial_V_errors(trial) = NaN;
        end
    end

    % avg and std deviation
    W_error(i) = mean(trial_W_errors, 'omitnan');
    V_error(i) = mean(trial_V_errors, 'omitnan');
    W_error_stds(i) = std(trial_W_errors, 'omitnan');
    V_error_stds(i) = std(trial_V_errors, 'omitnan');
end

output_folder = 'Thesis_code/experiments_plots_MC';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% V plot
figure;
errorbar(step_sizes, V_error, V_error_stds, '-o', 'LineWidth', 2, 'MarkerSize', 8);
set(gca, 'XScale', 'log', 'YScale', 'log', 'XDir', 'reverse');
xlabel('Step size (h)'); ylabel('Relative error in V');
title(sprintf('V Factor Matrix Error vs Step Size MC using Polynomial Fit\nfor f = [%s]', ...
    strjoin(arrayfun(@char, f, 'UniformOutput', false), '; ')));
grid on;
saveas(gcf, fullfile(output_folder, sprintf('factor_matrix_V_vs_stepsize_poly_%MC_f_id_%d.png', f_id)));

% W plot
figure;
errorbar(step_sizes, W_error, W_error_stds, '-o', 'LineWidth', 2, 'MarkerSize', 8);
set(gca, 'XScale', 'log', 'YScale', 'log', 'XDir', 'reverse');
xlabel('Step size (h)'); ylabel('Relative error in W');
title(sprintf('W Factor Matrix Error vs Step Size MC using Polynomial Fit\nfor f = [%s]', ...
    strjoin(arrayfun(@char, f, 'UniformOutput', false), '; ')));
grid on;
saveas(gcf, fullfile(output_folder, sprintf('factor_matrix_W_vs_stepsize_poly_f_id_%d.png', f_id)));

end
