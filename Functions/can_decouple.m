%% Check uniqueness condition for CPD decoupling
% m: number of input variables
% n: number of output functions
% r: CP rank (number of rank-1 terms)
% Output:
% canDecouple: true if uniqueness condition holds
% lhs: value of m(m−1) * n(n−1)
% rhs: value of 2 * r(r−1)
function [canDecouple, lhs, rhs] = can_decouple(m, n, r)

    lhs = m * (m - 1) * n * (n - 1);
    rhs = 2 * r * (r - 1);

    canDecouple = lhs >= rhs;
end
