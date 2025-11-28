clc;
clear;
ldata=[1 2 0.05 0.15 1 0.04
1 3 0.1 0.3 1 0.06
2 3 0.15 0.45 0.988 0
2 4 0.1 0.3 0.957 0
3 4 0.05 0.15 1 0.02];

fb = ldata(:,1);
tb = ldata(:,2);
R = ldata(:,3);
X = ldata(:,4);
a = ldata(:,5);
sh = ldata(:,6);

z = R + 1i*X;
nbus = max(max(fb), max(tb));
nbranch = length(fb);
ybus = zeros(nbus, nbus);

for m = 1:nbranch
    ybus(fb(m), tb(m)) = -1/(z(m)*a(m));
    ybus(tb(m), fb(m)) = -1/(z(m)*a(m));
end

for m = 1:nbranch
    ybus(fb(m), fb(m)) = ybus(fb(m), fb(m)) + 1/(z(m)*(a(m)^2)) + sh(m)/2;
    ybus(tb(m), tb(m)) = ybus(tb(m), tb(m)) + 1/(z(m)*(a(m)^2)) + sh(m)/2;
end

disp(ybus);


