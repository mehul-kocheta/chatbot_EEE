clear all;
clc;
%Busno Type vMag theta pGen qGen pLoad qLoad qMin qMax GshPu BshPu
baseMVA = 100;

loadData = [ 1, 1.05, 0,  0,     0;
            2, 1.00, 0, -0.4, -0.2;
            3, 1.02, 0,  0.3,   0 ];

ldata = [ 1, 2, 0.03, 0.08, 1, 0.04;
          1, 3, 0.02, 0.05, 1, 0.02;
          2, 3, 0.01, 0.03, 1, 0.03 ];

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

v_initial = loadData(:,2);
theta_initial = loadData(:,3);
bus_type = loadData(:,1);

v_cal = v_initial.*(cos(theta_initial) - 1i*sin(theta_initial));
v_cal_prev = v_cal;
v_cal_mag = abs(v_cal);
v_cal_angle = angle(v_cal);

p_spec = loadData(:,4)/baseMVA;
q_spec = loadData(:,5)/baseMVA;

%Ignoring the Qlimits
qmin = -999999*ones(nbus);
qmax = 9999999*ones(nbus);

nbus = length(ybus(:,1));

for k=1:100
    for m=1:nbus
        if bus_type(m) == 2
            q_cal = sum(v_cal_mag(m) * v_cal_mag.' .* abs(ybus(m,:)) .* sin(v_cal_angle(m) - v_cal_angle.' - angle(ybus(m,:))));
            if q_cal > qmax(m)
                q_cal = qmax(m);
            elseif q_cal < qmin(m)
                q_cal = qmin(m);
            end

            p_cal = p_spec(m) - 1i*q_cal;
            v_cal(m) = (1/ybus(m,m))*(p_cal/conj(v_cal(m)) - ((ybus(m,:) * v_cal) - ybus(m,m)*v_cal(m)));
        elseif bus_type(m) == 3
            p_cal = p_spec(m) - 1i*q_spec(m);
            v_cal(m) = (1/ybus(m,m))*(p_cal/conj(v_cal(m)) - ((ybus(m,:) * v_cal) - ybus(m,m)*v_cal(m)));
        end
    end

    diff = abs(v_cal_prev - v_cal);
    maxerr = max(diff);

    if maxerr < 0.0001
        break
    end

    v_cal_prev = v_cal;
    v_cal_mag = abs(v_cal);
    v_cal_angle = angle(v_cal);
end

disp(v_cal_mag)
disp(v_cal_angle)