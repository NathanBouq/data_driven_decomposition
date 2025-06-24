%% Recover scalar functions g_i from CPD derivative mode
% V: m*r factor matrix from CPD 
% H: N*r matrix of derivative samples g_i' values
% X: m*N matrix of sample input points
% Output:
% g_func: function handle that maps r*N matrix Z to n*N output matrix g(Z)
function g_func = recover_g_from_H(V, H, X)

    Z = V' * X; % project input points
    r = size(V, 2);
    g_fits = cell(1, r);
    
    for k = 1:r
        z_k = Z(k, :);
        g_k_prime = H(:, k)';
        
        [z_sorted, idx] = sort(z_k);
        g_prime_sorted = g_k_prime(idx);
        
        g_k = cumtrapz(z_sorted, g_prime_sorted);
        g_fits{k} = @(z) interp1(z_sorted, g_k, z, 'linear', 'extrap');
    end
   
    g_func = @(Z) cell2mat(arrayfun(@(k) g_fits{k}(Z(k, :)), 1:r, 'UniformOutput', false)');
end
