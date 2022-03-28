data = importdata("./hw2files/occam1.mat");
x = data.x;
y = data.y;




k = kernel(2);
disp(k);
syms t s;
f = t^2;
minimize(-1, f, 10, 10)
% [x_optimization,f_optimization] = Quasi_Newton_BFGS_Method(f,[-1 1],[t s]);
% x_optimization = double(x_optimization);
% f_optimization = double(f_optimization);
% x_optimization
% f_optimization








function k = kernel(x)
    k = [1 x x^2 x^3 x^4]; 
end
