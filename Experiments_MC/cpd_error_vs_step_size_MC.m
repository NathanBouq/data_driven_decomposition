%% CPD error vs step size using finite difference methods with Monte Carlo
% f: symbolic  function
% vars: symbolic variables
% X: m*N matrix of fixed sample input points
% r: CP rank for CPD
% f_id: function identifier
% methods: function handles 
% method_names: method name strings
% num_trials: number of Monte Carlo trials
% Output:
% errors: [length(step_sizes) x num_methods] CPD error matrix
function errors = cpd_error_vs_step_size_MC(f, vars, X, r, f_id, methods, method_names, num_trials)

step_sizes = logspace(-5, -2, 4); 
N = size(X, 2);
m = length(vars); 
errors = zeros(length(step_sizes), length(methods));    
error_stds = zeros(length(step_sizes), length(methods));  

f_num = matlabFunction(f, 'Vars', {vars(:)});

for j = 1:length(methods)
    diff_method = methods{j};

    for i = 1:length(step_sizes)
        h = step_sizes(i);
        trial_errors = zeros(1, num_trials);

        for trial = 1:num_trials
            try
                X = sample_input_points(m, N, f_id);
                Jo_tensor = compute_symbolic_jacobian_tensor(f, vars, X);
                [U_ref, ~] = cpd(Jo_tensor, r);

                J_est = diff_method(f_num, X, h);
                [U_est, ~] = cpd(J_est, r);

                trial_errors(trial) = norm(cpderr({U_ref{1}, U_ref{2}}, {U_est{1}, U_est{2}}));

            catch ME
                warning("Error at h=%.1e in %s (trial %d): %s", h, method_names{j}, trial, ME.message);
                trial_errors(trial) = NaN;
            end
        end

        % avg the CPD errors across all trials
        errors(i, j) = mean(trial_errors, 'omitnan');
        error_stds(i, j) = std(trial_errors, 'omitnan');
    end
end

output_folder = 'Thesis_code/experiments_plots_MC';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

ymin = 10^floor(log10(min(errors(:))));
ymax = 10^ceil(log10(max(errors(:))));
if ymax == max(errors(:))
    ymax = 10^(ceil(log10(ymax)) + 1);  % bump it up if it's tight
end

% one plot per differentiation method
for j = 1:length(methods)
    figure;

    errorbar(step_sizes, errors(:, j), error_stds(:, j), '-o', 'LineWidth', 2, 'MarkerSize', 8);
    set(gca, 'XScale', 'log', 'YScale', 'log', 'XDir', 'reverse');

    xlabel('Step size (h)');
    ylabel('Avg CPD Error');
    ylim([ymin, ymax]); % y-axis limits

    f_str = strjoin(arrayfun(@char, f, 'UniformOutput', false), '; ');
    title(sprintf('CPD Error vs Step Size MC using %s Method\nfor f = [%s]', ...
        method_names{j}, f_str));
    grid on;

    filename = sprintf('cpd_step_size_vs_error_%s_f_id_%d.png', lower(method_names{j}), f_id);
    saveas(gcf, fullfile(output_folder, filename));
end

end
