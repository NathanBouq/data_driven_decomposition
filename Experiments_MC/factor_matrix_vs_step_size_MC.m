%% CPD factor matrix error vs step size with Monte Carlo
% f: symbolic function
% vars: symbolic variables
% X: m*N matrix of input points
% r: CP rank for CPD
% f_id: function identifier
% methods: cell array of function handles 
% method_names: cell array of method name strings
% num_trials: number of Monte Carlo trials
function factor_matrix_vs_step_size_MC(f, vars, X, r, f_id, methods, method_names, num_trials)

step_sizes = logspace(-5, -2, 4);  
m = length(vars);  
f_num = matlabFunction(f, 'Vars', {vars(:)});
N = size(X, 2); 

V_error = zeros(length(step_sizes), length(methods));
W_error = zeros(length(step_sizes), length(methods));
V_error_stds = zeros(length(step_sizes), length(methods));
W_error_stds = zeros(length(step_sizes), length(methods));

for j = 1:length(methods)
    diff_method = methods{j};

    for i = 1:length(step_sizes)
        h = step_sizes(i);
        trial_V_errors = zeros(1, num_trials);
        trial_W_errors = zeros(1, num_trials);

        for trial = 1:num_trials
            try
                X = sample_input_points(m, N, f_id);
                Jo_tensor = compute_symbolic_jacobian_tensor(f, vars, X);
                [Uo, ~] = cpd(Jo_tensor, r);
                Wo = Uo{1};
                Vo = Uo{2};

                J_est = diff_method(f_num, X, h);
                [U_est, ~] = cpd(J_est, r);
                W_est = U_est{1};
                V_est = U_est{2};

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

                trial_W_errors(trial) = norm(Wo - W_est, 'fro')/ norm(Wo, 'fro');
                trial_V_errors(trial) = norm(Vo - V_est, 'fro')/ norm(Vo, 'fro');

            catch ME
                warning("Failed at h = %.1e using %s (trial %d): %s", h, method_names{j}, trial, ME.message);
                trial_W_errors(trial) = NaN;
                trial_V_errors(trial) = NaN;
            end
        end

        % avg and std deviation across trials
        W_error(i, j) = mean(trial_W_errors, 'omitnan');
        V_error(i, j) = mean(trial_V_errors, 'omitnan');
        W_error_stds(i, j) = std(trial_W_errors, 'omitnan');
        V_error_stds(i, j) = std(trial_V_errors, 'omitnan');
    end
end

output_folder = 'Thesis_code/experiments_plots_MC';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

yminV = 10^floor(log10(min(V_error(:))));
ymaxV = 10^ceil(log10(max(V_error(:))));
if ymaxV == max(V_error(:))
    ymaxV = 10^(ceil(log10(ymaxV)) + 1);  
end

yminW = 10^floor(log10(min(W_error(:))));
ymaxW = 10^ceil(log10(max(W_error(:))));
if ymaxW == max(W_error(:))
    ymaxW = 10^(ceil(log10(ymaxW)) + 1); 
end

% plot of V factor matrix error vs step size 
for j = 1:length(methods)
    figure;
    errorbar(step_sizes, V_error(:, j), V_error_stds(:, j), '-o', 'LineWidth', 2, 'MarkerSize', 8);
    set(gca, 'XScale', 'log', 'YScale', 'log', 'XDir', 'reverse');
    xlabel('Step size (h)');
    ylabel('Relative error in V');
    % ylim([yminV, ymaxV]);

    f_str = strjoin(arrayfun(@char, f, 'UniformOutput', false), '; ');
    title(sprintf('V Factor Matrix Error vs Step Size MC using %s Method\nfor f = [%s]', ...
        method_names{j}, f_str));
    
    grid on;
    filename = sprintf('factor_matrix_V_vs_stepsize_%s_MC.png', lower(method_names{j}));
    saveas(gcf, fullfile(output_folder, filename));
end

% plot of W factor matrix error vs step size
for j = 1:length(methods)
    figure;
    errorbar(step_sizes, W_error(:, j), W_error_stds(:, j), '-o', 'LineWidth', 2, 'MarkerSize', 8);
    set(gca, 'XScale', 'log', 'YScale', 'log', 'XDir', 'reverse');
    xlabel('Step size (h)');
    ylabel('Relative error in W');
    % ylim([yminW, ymaxW]);
    
    f_str = strjoin(arrayfun(@char, f, 'UniformOutput', false), '; ');
    title(sprintf('W Factor Matrix Error vs Step Size MC using %s Method\nfor f = [%s]', ...
        method_names{j}, f_str));
    
    grid on;
    filename = sprintf('factor_matrix_W_vs_stepsize_%s_f_id_%d.png', lower(method_names{j}), f_id);
    saveas(gcf, fullfile(output_folder, filename));
end
end
