%% Compute symbolic Jacobian and evaluate at a point
% f: symbolic function
% vars: symbolic variable vector
% point: m*1 vector at which to evaluate the Jacobian
% Output:
% J: symbolic Jacobian matrix
% J_point: numerical Jacobian evaluated at the given point
function [J, J_point] = compute_symbolic_jacobian(f, vars, point)
    
    J = jacobian(f, vars);
    J_point = double(subs(J, vars, point));
end