%% CPD error vs number of sample points with Monte Carlo
% f: symbolic function
% vars: symbolic variables
% r: CP rank for CPD
% f_id: function identifier
% methods: function handles for Jacobian estimation methods
% method_names: method name strings (for labeling/plotting)
% num_trials: number of Monte Carlo trials per sample size
% Output:
% (No return value) — generates plot of CPD error vs N for all methods
function cpd_error_vs_npoints_MC(f, vars, r, f_id, methods, method_names, num_trials)

step_size = 1e-5;
Ns = round(logspace(2, 3.2, 7));
m = length(vars);   

f_num = matlabFunction(f, 'Vars', {vars(:)});

errors = zeros(length(Ns), length(method_names));      
error_stds = zeros(length(Ns), length(method_names));

for k = 1:length(methods)
    diff_method = methods{k};

    for i = 1:length(Ns)
        N = Ns(i);
        trial_errors = zeros(1, num_trials);

        for trial = 1:num_trials
            try
                X = sample_input_points(m, N, f_id);
                Jo_tensor = compute_symbolic_jacobian_tensor(f, vars, X);
                [Uo, ~] = cpd(Jo_tensor, r);

                J_tensor = diff_method(f_num, X, step_size);
                [U, ~] = cpd(J_tensor, r);

                trial_errors(trial) = norm(cpderr({Uo{1}, Uo{2}}, {U{1}, U{2}}));
                
            catch ME
                warning("Failed at N = %d using %s (trial %d): %s", N, method_names{k}, trial, ME.message);
                trial_errors(trial) = NaN;
            end
        end
        errors(i, k) = mean(trial_errors, 'omitnan');
        error_stds(i, k) = std(trial_errors, 'omitnan');
    end
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

% one plot per method
for j = 1:length(methods)
    figure;
    errorbar(Ns, errors(:, j), error_stds(:, j), '-o', 'LineWidth', 2, 'MarkerSize', 8);
    set(gca, 'XScale', 'log', 'YScale', 'log');
    xlabel('Number of sample points (N)');
    ylabel('Avg CPD Error');
    ylim([ymin, ymax]);
    
    f_str = strjoin(arrayfun(@char, f, 'UniformOutput', false), '; ');
    title(sprintf('CPD Error vs Number of Sample Points MC using %s Method\nfor f = [%s]', ...
        method_names{j}, f_str));
    grid on;

    filename = sprintf('cpderror_vs_npoints_%s_f_id_%d.png', lower(method_names{j}), f_id);
    saveas(gcf, fullfile(output_folder, filename));
end

end
