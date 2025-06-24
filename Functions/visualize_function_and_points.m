%% Visualize function output and sample points in 3D
% f: symbolic function 
% X: m*N matrix of input points
% vars: symbolic variables
% Output:
% 3D scatter plot of inputs and f(X)
function visualize_function_and_points(f, X, vars)

    f_num = matlabFunction(f, 'Vars', {vars(:)});
    n = length(f_num(rand(size(X,1),1)));
    N = size(X, 2);

    f_vals = zeros(n, N);
    for i = 1:N
        f_vals(:, i) = f_num(X(:, i));
    end

    % color = norm of f(x) 
    color_vals = vecnorm(f_vals); 

    figure;
    scatter3(X(1, :), X(2, :), X(3, :), 25, color_vals, 'filled');
    colorbar;

    f_str = strjoin(arrayfun(@char, f, 'UniformOutput', false), '; ');
    title(sprintf('Function values over sampled points for f = [%s]', ...
        f_str));
    xlabel('x'); ylabel('y'); zlabel('z');
    axis equal; grid on;

    output_folder = 'Thesis_code/experiments_plots_MC';
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end
    filename = 'function_visualization.png';
    saveas(gcf, fullfile(output_folder, filename));
end

