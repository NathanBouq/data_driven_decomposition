%% Visualize true vs. decoupled function outputs
% f_num: numerical function
% X: m*N matrix of input points
% f_approx_vals: n*N matrix of approximated output values
% f_sym: symbolic function 
% method_name: string of method name
% Output:
% f_true_vals: n*N matrix of true output values f(X)
% saves scatter plot to: Thesis_code/visualization_plot/visualization.png
function f_true_vals = visualize_decoupling(f_num, X, f_approx_vals, f_sym, method_name)

    output_folder = 'Thesis_code/visualization_plot';
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

    f_true_vals = f_num(X);

    scatter(f_true_vals(:), f_approx_vals(:), 10, 'filled');
    xlabel('f(x) true'); 
    ylabel('f(x) approx');

    f_str = strjoin(arrayfun(@char, f_sym, 'UniformOutput', false), ', ');
    title({sprintf('True vs Decoupled Output for %s', method_name), ...
       sprintf('f(x) = [%s]', f_str)});
    axis equal; grid on;

    filename = sprintf('visualization.png');
    saveas(gcf, fullfile(output_folder, filename));
end
