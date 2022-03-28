function [x_optimization,f_optimization] = Quasi_Newton_BFGS_Method(f,x0,var_x,epsilon)
format long;
%   f：目标函数
%   x0：初始点
%   var_x：自变量向量
%   epsilon：精度
%   x_optimization：目标函数取最小值时的自变量值
%   f_optimization：目标函数的最小值
if nargin == 3
    epsilon = 1.0e-6;
end
x0 = transpose(x0);
var_x = transpose(var_x);
n = length(var_x);
syms t;
H0 = eye(n,n);
grad_fx = jacobian(f,var_x);
grad_fx0  = subs(grad_fx,var_x,x0);
p0 = - H0 * transpose(grad_fx0);
k = 0;
xk = x0;
pk = p0;
Hk = H0;

while 1
    grad_fxk  = subs(grad_fx,var_x,xk);
    if norm(grad_fxk) <= epsilon
        x_optimization = x0;
        break;
    end   
    f_xk_tkpk = subs(f,var_x,(xk + t*pk));
    [xmin,xmax] = Advance_and_Retreat_Method(f_xk_tkpk,0,0.00001);
    tk = Golden_Selection_Method(f_xk_tkpk,xmin,xmax);    
    xk_next = xk + tk*pk;    
    grad_fxk_next = subs(grad_fx,var_x,xk_next);
    if norm(grad_fxk_next) <= epsilon
        x_optimization = xk_next;
        break;
    end
    if k + 1 == n
        xk = xk_next;
        continue;
    else
        Sk = xk_next - xk;
        yk = grad_fxk_next - grad_fxk;
        yk = transpose(yk);        
        Sk_T = transpose(Sk);
        yk_T = transpose(yk);
        m_Sk = Sk*Sk_T;
        Hk_next = Hk + (m_Sk/(Sk_T*yk))*(1 + (yk_T*Hk*yk)/(Sk_T*yk)) - (Sk*yk_T*Hk + Hk*yk*Sk_T)/(Sk_T*yk);
        pk = - Hk_next * transpose(grad_fxk_next);
        k = k + 1;
        xk = xk_next;
        Hk = Hk_next;
    end
end
f_optimization = subs(f,var_x,x_optimization);
format short;