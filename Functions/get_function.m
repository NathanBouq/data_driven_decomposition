%% Retrieve predefined symbolic function by difficulty level
% difficulty: integer selecting the function, easy = 1, medium = 3, hard =
% 3
% Output:
% f: symbolic function
% vars: symbolic variables used in f
% r: CP rank used for CPD 
% degree: degree of the polynomial function
function [f, vars, r, degree] = get_function(difficulty)

    switch difficulty
        case 1
            % polynomial function
            syms x y
            f = [ (x + y)^2; (x - y)^3 ];
            vars = [x; y];
            r = 2;
            degree = 3;
        case 2
            % medium function
            syms x y z
            f = [ sin(x + y) + z^2; x*y*z + cos(z) ];
            vars = [x; y; z];
            r = 3;
            degree = 4;
        case 3
            % more complex nonlinear function
            syms x y z
            f = [ exp(x*y) + sin(z); log(1 + x^2 + y^2); (x + y + z)^3 ];
            vars = [x; y; z];
            r = 4;
            degree = 4;
    end
end
