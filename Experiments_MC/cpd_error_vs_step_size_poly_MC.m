%% CPD error vs step size using polynomial regression with Monte Carlo
% f: symbolic function
% vars: symbolic variables
% X: m*N matrix of fixed input points
% r: CP rank for CPD
% f_id: function identifier 
% num_trials: number of Monte Carlo trials per step size
% degree: degree of the polynomial regression
% Output:
% errors: [length(step_sizes) x 1] vector of average CPD errors
function errors = cpd_error_vs_step_size_poly_MC(f, vars, X, r, f_id, num_trials, degree)

step_sizes = logspace(-5, -2, 4);  
N = size(X, 2);
m = length(vars);

f_num = matlabFunction(f, 'Vars', {vars(:)});

errors = zeros(length(step_sizes), 1);
error_stds = zeros(length(step_sizes), 1);

for i = 1:length(step_sizes)
    h = step_sizes(i);
    trial_errors = zeros(1, num_trials);

    for trial = 1:num_trials
        try
            X = sample_input_points(m, N, f_id);
            Y = f_num(X);
            [f_poly, J_est] = fit_polynomial_model(X, Y, degree);

            Jo_tensor = compute_symbolic_jacobian_tensor(f, vars, X);
            [U_ref, ~] = cpd(Jo_tensor, r);

            [U_est, ~] = cpd(J_est, r);

            trial_errors(trial) = norm(cpderr({U_ref{1}, U_ref{2}}, {U_est{1}, U_est{2}}));

        catch ME
            warning("Error at h=%.1e in %s (trial %d): %s", h, method_name, trial, ME.message);
            trial_errors(trial) = NaN;
        end
    end

    % avg the CPD errors across all trials
    errors(i) = mean(trial_errors, 'omitnan');
    error_stds(i) = std(trial_errors, 'omitnan');
end

output_folder = 'Thesis_code/experiments_plots_MC';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

ymin = 10^floor(log10(min(errors(:))));
ymax = max(10^ceil(log10(max(errors(:)))), 1e-8);
if ymax == max(errors(:))
    ymax = 10^(ceil(log10(ymax)) + 1);  % bump it up if it's tight
end

% plot
figure;
errorbar(step_sizes, errors(:), error_stds(:), '-o', 'LineWidth', 2, 'MarkerSize', 8);
set(gca, 'XScale', 'log', 'YScale', 'log', 'XDir', 'reverse');
% ylim([ymin, ymax]);

xlabel('Step size (h)');
ylabel('Avg CPD Error');
f_str = strjoin(arrayfun(@char, f, 'UniformOutput', false), '; ');
title(sprintf('CPD Error vs Step Size MC using Polynomial Fit\nfor f = [%s]', ...
    f_str));
grid on;

filename = sprintf('cpd_step_size_vs_error_poly_f_id_%d.png', f_id);
saveas(gcf, fullfile(output_folder, filename));

end
