%% Recover symbolic functions g_i using global polynomial regression
% W: n*r factor matrix from CPD 
% V: m*r factor matrix from CPD 
% X: m*N matrix of input points
% Y: n*N matrix of function outputs f(X)
% d: degree of the polynomial models to fit
% Output:
% G: r×1 symbolic vector of recovered scalar functions g_i(u_i)
function G = recover_g_from_vandermonde(W, V, X, Y, d)

    [~, N] = size(X);
    [n, ~] = size(Y);
    r = size(V, 2);
    
    Z = V' * X; 
    
    block_diagonal_W = zeros(N * n, N * r); % diagonal matrix W
    for i = 0:N-1
        block_diagonal_W(1+i*n:i*n+n, 1+i*r:i*r+r) = W;
    end
    
    Xk = zeros(N*r, r*(d+1)); % matrix for polynomial terms
    for k = 1:N
        for j = 1:r
            for i = 0:d
                Xk((k-1)*r + j, i+1 + (j-1)*(d+1)) = Z(j, k)^i;
            end
        end
    end
    
    Zvec = reshape(Y, N*n, 1); % stack columns of Y
    C = pinv(block_diagonal_W * Xk) * Zvec;
    
    % build symbolic g_i(u_i) functions
    syms u [1 r]
    G = sym(zeros(r, 1));
    for j = 1:r
        expr = 0;
        for i = 0:d
            idx = i+1 + (j-1)*(d+1);
            expr = expr + C(idx) * u(j)^i;
        end
        G(j) = simplify(expr);
    end
end
