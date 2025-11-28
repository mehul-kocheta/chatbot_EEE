clc;
clear;

%% ----------------------- INPUT DATA -----------------------
% Y-bus matrix (admittance matrix)
Y = [  3 - 8.95j,   -2 + 6j,     -1 + 3j,     0;
       -2 + 6j,      3.774 - 11.306j,   -0.674 + 2.024j,  -1.044 + 3.134j;
       -1 + 3j,      -0.674 + 2.024j,    3.666 - 10.96j,  -2 + 6j;
        0,            -1.044 + 3.134j,   -2 + 6j,          3 - 8.99j ];

Ymag   = abs(Y);       % Magnitude of admittance
ThetaY = angle(Y);     % Angle of admittance

% Bus data: [BusNo, |V|, δ, P_spec, Q_spec]
busData = [ 1, 1.05, 0,    0,     0;
            2, 1.00, 0,    0.45,  0.22;
            3, 1.00, 0,   -0.98,  0.40;
            4, 1.00, 0,    0.32, -0.15 ];

p_spec = busData(:,4);
q_spec = busData(:,5);
v_mag = busData(:,2);
v_ang = busData(:,3);

nbus = length(v_mag);
p_cal = zeros(nbus,1);
q_cal = zeros(nbus,1);

tol = 0.2;

iteration = 0;

while tol > 1e-4

    p_cal = zeros(nbus,1);
    q_cal = zeros(nbus,1);

    for i=1:nbus
        for j=1:nbus
            p_cal(i) = p_cal(i) + v_mag(i)*v_mag(j)*Ymag(i,j)*cos(v_ang(i) - v_ang(j) - ThetaY(i,j));
            q_cal(i) = q_cal(i) + v_mag(i)*v_mag(j)*Ymag(i,j)*sin(v_ang(i) - v_ang(j) - ThetaY(i,j));
        end
    end

    p_diff = p_spec - p_cal;
    q_diff = q_spec - q_cal;

    mismatch = [p_diff(2:nbus,:) ; q_diff(2:nbus,:)];
    disp(mismatch)
    tol = max(abs(mismatch));
    
    %J1 = delP/del(delta)
    J1 = zeros(nbus-1, nbus-1);
    for i=2:nbus
        for j=2:nbus
            if i==j
                J1(i-1,j-1) = -1*q_cal(i) - (v_mag(i)^2)*(Ymag(i,j)*sin(ThetaY(i,j)));
            else
                J1(i-1,j-1) = v_mag(i)*v_mag(j)*Ymag(i,j)*sin(v_ang(i) - v_ang(j) - ThetaY(i,j));
            end
        end
    end

    J2 = zeros(nbus-1,nbus-1);
    for i=2:nbus
        for j=2:nbus
            if i==j
                J2(i-1,j-1) = (p_cal(i)/v_mag(i)) + v_mag(i)*Ymag(i,j)*cos(ThetaY(i,j));
            else
                J2(i-1,j-1) = v_mag(i)*Ymag(i,j)*cos(v_ang(i)-v_ang(j)-ThetaY(i,j));
            end
        end
    end

    Q1 = zeros(nbus-1,nbus-1);
    for i=2:nbus
        for j=2:nbus
            if i==j
                Q1(i-1,j-1) = p_cal(i) - (v_mag(i)^2)*Ymag(i,j)*cos(ThetaY(i,j));
            else
                Q1(i-1,j-1) = -1*v_mag(i)*v_mag(j)*Ymag(i,j)*cos(v_ang(i)-v_ang(j)-ThetaY(i,j));
            end
        end
    end

    Q2 = zeros(nbus-1,nbus-1);
    for i=2:nbus
        for j=2:nbus
            if i==j
                Q2(i-1,j-1) = (q_cal(i)/v_mag(i)) - v_mag(i)*Ymag(i,i)*sin(ThetaY(i,i));
            else
                Q2(i-1,j-1) = v_mag(i)*Ymag(i,j)*sin(v_ang(i)-v_ang(j)-ThetaY(i,j));
            end
        end
    end

    J = [J1 J2; Q1 Q2];
    disp(J)

    X = J \ mismatch;

    disp(X)

    dDelta = X(1:nbus-1);
    dVmag  = X(nbus:end);

    v_mag(2:nbus)  = v_mag(2:nbus)  + dVmag;
    v_ang(2:nbus) = v_ang(2:nbus) + dDelta;

    iteration = iteration + 1;

    % Display iteration 1 values
    if iteration == 1
        fprintf('\nIteration %d\n', iteration);
        fprintf('Bus Voltages (pu):\n');
        disp(v_mag);
        fprintf('Bus Angles (radians):\n');
        disp(v_ang);
        %break
    end
end

disp(v_mag)
disp(v_ang)