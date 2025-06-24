%% 
%% Average Jacobian slice error vs step size using polynomial regression with Monte Carlo
% f: symbolic function
% vars: symbolic variables
% X_init: m*N matrix of initial input points
% r: CP rank for CPD 
% f_id: function identifier 
% num_trials: number of Monte Carlo trials 
% degree: degree of the polynomial regression
% Output:
% errors: [length(step_sizes) x 1] vector of average Frobenius norm errors
% between estimated and symbolic Jacobian slices
function errors = jacobian_slice_errors_poly_MC(f, vars, X_init, r, f_id, num_trials, degree)

step_sizes = logspace(-5, -2, 4);
N = size(X_init, 2);
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
            [f_poly, J_approx] = fit_polynomial_model(X, Y, degree);

            Jo_tensor = compute_symbolic_jacobian_tensor(f, vars, X);

        catch ME
            warning("Failed for method %s at h=%.1e (trial %d): %s", ...
                method_name, h, trial, ME.message);
            trial_errors(trial) = NaN;
            continue;
        end

        slice_errors = zeros(1, N);
        for k = 1:N
            slice_errors(k) = norm(Jo_tensor(:, :, k) - J_approx(:, :, k), 'fro') / norm(Jo_tensor(:, :, k), 'fro');
        end
        trial_errors(trial) = mean(slice_errors);
    end

    errors(i) = mean(trial_errors, 'omitnan'); 
    error_stds(i) = std(trial_errors, 'omitnan');
end

output_folder = 'Thesis_code/experiments_plots_MC';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

ymin = 10^floor(log10(min(errors(:))));
ymax = 10^ceil(log10(max(errors(:))));
if ymax == max(errors(:))
    ymax = 10^(ceil(log10(ymax)) + 1); 
end

% Plot
figure;
errorbar(step_sizes, errors(:), error_stds(:), '-o', 'LineWidth', 2, 'MarkerSize', 8);
set(gca, 'XScale', 'log', 'YScale', 'log', 'XDir', 'reverse');
% ylim([ymin, ymax]);

xlabel('Step size (h)');
ylabel('Avg Mean relative Jacobian slice error');
f_str = strjoin(arrayfun(@char, f, 'UniformOutput', false), '; ');
title(sprintf('Slice Error vs Step Size MC using Polynomial Fit\nfor f = [%s]', ...
    f_str));
grid on;

filename = sprintf('jacobian_slice_vs_stepsize_poly_f_id_%d.png', f_id);
saveas(gcf, fullfile(output_folder, filename));

end
