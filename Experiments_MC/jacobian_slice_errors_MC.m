%% Average Jacobian slice error vs step size witu Monte Carlo
% f: symbolic function
% vars: symbolic variables
% X: m*N matrix of input points
% r: CP rank for CPD 
% f_id: function identifier 
% methods: finite difference Jacobian estimation methods
% method_names: method names 
% num_trials: number of Monte Carlo trials
% Output:
% errors: [length(step_sizes) x num_methods] matrix of average Frobenius norm errors
% between estimated Jacobian slices and symbolic Jacobians
function errors = jacobian_slice_errors_MC(f, vars, X, r, f_id, methods, method_names, num_trials)

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
                J_approx = diff_method(f_num, X, h);
            catch ME
                warning("Failed for method %s at h=%.1e (trial %d): %s", ...
                    method_names{j}, h, trial, ME.message);
                trial_errors(trial) = NaN;
                continue;
            end

            slice_errors = zeros(1, N);
            for k = 1:N
                slice_errors(k) = norm(Jo_tensor(:, :, k) - J_approx(:, :, k), 'fro') / norm(Jo_tensor(:, :, k), 'fro');
            end
            trial_errors(trial) = mean(slice_errors);
        end
        
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
    ymax = 10^(ceil(log10(ymax)) + 1);  % +1 so that it's not on the x-axis
end

for j = 1:length(methods)
    figure;
    
    errorbar(step_sizes, errors(:, j), error_stds(:, j), '-o', 'LineWidth', 2, 'MarkerSize', 8);
    set(gca, 'XScale', 'log', 'YScale', 'log', 'XDir', 'reverse');  

    xlabel('Step size (h)');
    ylabel('Avg Mean relative Jacobian slice error');
    % ylim([ymin, ymax]);

    f_str = strjoin(arrayfun(@char, f, 'UniformOutput', false), '; ');
    title(sprintf('Slice Error vs Step Size MC using %s Method\nfor f = [%s]', ...
        method_names{j}, f_str));
    grid on;

    filename = sprintf('jacobian_slice_vs_stepsize_%s_f_id_%d.png', lower(method_names{j}), f_id);
    saveas(gcf, fullfile(output_folder, filename));
end

end
