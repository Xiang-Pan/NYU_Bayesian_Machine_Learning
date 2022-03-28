E = numel(gps);
ns = size(xstar,1);

ystar = zeros(ns,E);
sigma_star = zeros(E,E);
temp = zeros(ns,E);

parfor i = 1:E
    [ystar(:,i), temp(:,i)] = gps(i).Regress(xstar);
end

if ns == 1
    sigma_star = diag(temp);
end
res.ystar = ystar;
res.sigma_star = sigma_star;
