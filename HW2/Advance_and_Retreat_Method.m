%   进退法
function [xmin,xmax] = Advance_and_Retreat_Method(f,x0,h0)
format long;
%   f：目标函数
%   x0：初始点
%   h0：初始步长
%   epsilon：精度
%   xmin：目标函数取包含极值的区间左端点
%   xmax：目标函数取包含极值的区间右端点
if nargin == 3
    epsilon = 1.0e-6;
end
%   【1】
h = h0;
x1 = x0;
k = 0;
while 1
    %   【2】
    x4 = x1 + h;    %   试探步
    k = k + 1;
    %   【3】
    fx4 = subs(f, symvar(f),x4);
    fx1 = subs(f, symvar(f),x1);
    if fx4 < fx1
    %   【4】
        x2 = x1;
        x1 = x4;
        fx2 = fx1;
        fx1 = fx4;
        h = 2*h;    %   加大步长
    else
        if k == 1   %   【5】
        %   【6】
            h = -h;     %   反向搜索
            x2 = x4;
            fx2 = fx4;
        else    %   【7】
            x3 = x2;
            x2 = x1;
            x1 = x4;
            break;
        end
    end
end
xmin = min(x1,x3);
xmax = x1 + x3 - xmin;
format short;