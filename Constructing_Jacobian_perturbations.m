addpath('Functions', 'Differentiation_methods');

%% Part 1: Constructing the Jacobian Tensor

% Define symbolic variables and function

syms x y z;
f_id = 3;
[f, vars, r] = get_function(f_id);
f_num = matlabFunction(f, 'Vars', {vars(:)});  
h = 1e-8; % step size
N = 1000; % number of sample points
[X, Y] = generate_data_points(f, vars, N);  % X: m x N, Y: n x N

n = length(f_num(X(:,1)));  % number of outputs
m = length(vars);       % number of inputs

% sanity check for decoupling identifiability
[ok, lhs, rhs] = can_decouple(m, n, r);

if ok
    disp("The uniqueness condition is satisfied. You can generically decouple f.");
else
    disp("The condition is not satisfied. You may still decouple, but it's not guaranteed.");
end

fprintf("LHS = %d, RHS = %d\n", lhs, rhs);

% Compute the Jacobian tensor using method of choice
method_name = ('Polynomial regression');
J_tensor = local_poly_regression(X, Y, 10, 2);
disp('Jacobian tensor constructed (dimensions: outputs x inputs x sample points):');
disp(size(J_tensor));

%% Part 2: CP Decomposition using Tensorlab

% estimate the rank to check using tensorlab
est_r = rankest(J_tensor);
disp("Estimated CP rank:");
disp(est_r);

% Perform CP decomposition
[U, output] = cpd(J_tensor, r);
W_J = U{1};
V_J = U{2};
H_J = U{3};

disp('Factor matrix for outputs (approximate W):');
disp(W_J);
disp('Factor matrix for inputs (approximate V):');
disp(V_J);
disp('Factor matrix for derivative evaluations (approximate H):');
% disp(H_J);

% Compute symbolic Jacobian tensor for ground truth
Jo_tensor = compute_symbolic_jacobian_tensor(f, vars, X);
[Uo, output2] = cpd(Jo_tensor, r);
Wo = Uo{1};
Vo = Uo{2};
Ho = Uo{3};

%% Part 3: Error checking
cpderrorJ = cpderr({Wo, Vo}, {W_J, V_J});
disp('The cpd error is:')
disp(cpderrorJ)

if norm(cpderrorJ) < 1e-4
    disp('Good result!');
else
    warning('There seems to be a problem; local minimum? Try to re-run decoupleJ.');
end

%% Part 4: Recover scalar functions g_i and visualize (symbolic method)
choice = 2;

if choice == 1
    g_func = recover_g_from_H(V_J, H_J, X);  % numerical function 
    f_approx_vals = W_J * g_func(V_J' * X);  % numerical evaluation

else
    d = 3;
    G = recover_g_from_vandermonde(W_J, V_J, X, Y, d);  % symbolic

    syms x [1 m]
    Z_sym = V_J' * x.';
    u = sym('u', [1 r]).';
    g_sym = subs(G, u, Z_sym);
    f_decoupled = W_J * g_sym; 

    f_decoupled_num = matlabFunction(f_decoupled, 'Vars', {x});

    % evaluate over all sample points
    f_approx_vals = zeros(n, N);
    for i = 1:N
        f_approx_vals(:, i) = f_decoupled_num(X(:, i).');
    end
end

% visualization
visualize_decoupling(f_num, X, f_approx_vals, f, method_name);
