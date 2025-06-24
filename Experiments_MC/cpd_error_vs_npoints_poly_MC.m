%% CPD error vs number of sample points using polynomial regression with Monte Carlo
% f: symbolic function
% vars: symbolic variables
% r: CP rank for CPD
% f_id: function identifier
% num_trials: number of Monte Carlo trials per N
% degree: degree of the polynomial regression
% Output:
% errors: [length(Ns) x 1] vector of average CPD errors across sample sizes
function errors = cpd_error_vs_npoints_poly_MC(f, vars, r, f_id, num_trials, degree)

step_size = 1e-5;
Ns = round(logspace(2, 3.2, 7));
m = length(vars); 

f_num = matlabFunction(f, 'Vars', {vars(:)});

errors = zeros(length(Ns), 1);     
error_stds = zeros(length(Ns), 1);

for i = 1:length(Ns)
    N = Ns(i);
    trial_errors = zeros(1, num_trials);

    for trial = 1:num_trials
        try
            X = sample_input_points(m, N, f_id);
            Y = f_num(X);
            [f_poly, J_tensor] = fit_polynomial_model(X, Y, degree);

            Jo_tensor = compute_symbolic_jacobian_tensor(f, vars, X);
            [Uo, ~] = cpd(Jo_tensor, r);

            [U, ~] = cpd(J_tensor, r);

            trial_errors(trial) = norm(cpderr({Uo{1}, Uo{2}}, {U{1}, U{2}}));

        catch ME
            warning("Failed at N = %d using %s (trial %d): %s", N, method_name, trial, ME.message);
            trial_errors(trial) = NaN;
        end
    end

    % avg and std deviation across trials
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

figure;
errorbar(Ns, errors(:), error_stds(:), '-o', 'LineWidth', 2, 'MarkerSize', 8);
set(gca, 'XScale', 'log', 'YScale', 'log');
% ylim([ymin, ymax]);

xlabel('Number of sample points (N)');
ylabel('Avg CPD Error');
f_str = strjoin(arrayfun(@char, f, 'UniformOutput', false), '; ');
title(sprintf('CPD Error vs Number of Sample Points MC using Polynomial Fit\nfor f = [%s]', ...
    f_str));
grid on;

filename = sprintf('cpderror_vs_npoints_poly_f_id_%d.png', f_id);
saveas(gcf, fullfile(output_folder, filename));

end
