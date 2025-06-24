%% Sample input points based on function type
% m: number of input variables
% N: number of sample points to generate
% f_id: function ID (1 = polynomial, 2/3 = sinusoidal)
% Output:
% X: m*N matrix of input points (domain depends on f_id)
function X = sample_input_points(m, N, f_id)

    if f_id == 1
        X = rand(m, N);  % uniform in [0, 1]
    elseif f_id == 2 || f_id == 3
        X = rand(m, N) * 1.0 - 0.5;  % uniform in [-0.5, 0.5]
    end
end
