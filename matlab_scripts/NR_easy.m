clc;
clear;

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


%% ----------------------- INPUT DATA -----------------------
% Y-bus matrix (admittance matrix)
Y = ybus;

Ymag   = abs(Y);       % Magnitude of admittance
ThetaY = angle(Y);     % Angle of admittance

% Bus data: [BusNo, |V|, δ, P_spec, Q_spec]
busData = [ 1, 1.05, 0,  0,     0;
            2, 1.00, 0, -0.4, -0.2;
            3, 1.02, 0,  0.3,   0 ];

% Extract columns
busNo = busData(:,1);
Vmag  = busData(:,2);
Delta = busData(:,3);
Psp   = busData(:,4);
Qsp   = busData(:,5);

nBus = max(busNo);

% Initialize variables
VmagNew  = Vmag;
DeltaNew = Delta;
iteration = 0;
tolerance = 2;   % start with a large tolerance

%% ----------------------- ITERATION LOOP -----------------------
while tolerance > 1e-4
    % --- Step 1: Calculate P and Q ---
    Pcal = zeros(nBus,1);
    Qcal = zeros(nBus,1);
    for i = 1:nBus
        for k = 1:nBus
            Pcal(i) = Pcal(i) + Vmag(i)*Ymag(i,k)*Vmag(k)*cos(ThetaY(i,k) - Delta(i) + Delta(k));
            Qcal(i) = Qcal(i) - Vmag(i)*Ymag(i,k)*Vmag(k)*sin(ThetaY(i,k) - Delta(i) + Delta(k));
        end
    end

    % --- Step 2: Mismatch vector ---
    dP = Psp - Pcal;
    dQ = Qsp - Qcal;
    mismatch = [dP(2:nBus); dQ(2:nBus)];
    disp(mismatch)
    tolerance = max(abs(mismatch));

    % --- Step 3: Form Jacobian submatrices ---
    % J1 = dP/dδ
    J1 = zeros(nBus-1, nBus-1);
    for i = 2:nBus
        for k = 2:nBus
            if i == k
                for b = 1:nBus
                    J1(i-1,k-1) = J1(i-1,k-1) + Vmag(i)*Ymag(i,b)*Vmag(b)*sin(ThetaY(i,b) - Delta(i) + Delta(b));
                end
                J1(i-1,k-1) = J1(i-1,k-1) - (Vmag(i)^2)*Ymag(i,i)*sin(ThetaY(i,i));
            else
                J1(i-1,k-1) = -Vmag(i)*Ymag(i,k)*Vmag(k)*sin(ThetaY(i,k) - Delta(i) + Delta(k));
            end
        end
    end

    % J2 = dP/d|V|
    J2 = zeros(nBus-1, nBus-1);
    for i = 2:nBus
        for k = 2:nBus
            if i == k
                for b = 1:nBus
                    J2(i-1,k-1) = J2(i-1,k-1) + Vmag(b)*Ymag(i,b)*cos(ThetaY(i,b) - Delta(i) + Delta(b));
                end
                J2(i-1,k-1) = J2(i-1,k-1) + Vmag(i)*Ymag(i,i)*cos(ThetaY(i,i));
            else
                J2(i-1,k-1) = Vmag(i)*Ymag(i,k)*cos(ThetaY(i,k) - Delta(i) + Delta(k));
            end
        end
    end

    % J3 = dQ/dδ
    J3 = zeros(nBus-1, nBus-1);
    for i = 2:nBus
        for k = 2:nBus
            if i == k
                for b = 1:nBus
                    J3(i-1,k-1) = J3(i-1,k-1) + Vmag(i)*Ymag(i,b)*Vmag(b)*cos(ThetaY(i,b) - Delta(i) + Delta(b));
                end
                J3(i-1,k-1) = J3(i-1,k-1) - (Vmag(i)^2)*Ymag(i,i)*cos(ThetaY(i,i));
            else
                J3(i-1,k-1) = -Vmag(i)*Ymag(i,k)*Vmag(k)*cos(ThetaY(i,k) - Delta(i) + Delta(k));
            end
        end
    end

    % J4 = dQ/d|V|
    J4 = zeros(nBus-1, nBus-1);
    for i = 2:nBus
        for k = 2:nBus
            if i == k
                for b = 1:nBus
                    J4(i-1,k-1) = J4(i-1,k-1) - Vmag(b)*Ymag(i,b)*sin(ThetaY(i,b) - Delta(i) + Delta(b));
                end
                J4(i-1,k-1) = J4(i-1,k-1) - Vmag(i)*Ymag(i,i)*sin(ThetaY(i,i));
            else
                J4(i-1,k-1) = -Vmag(i)*Ymag(i,k)*sin(ThetaY(i,k) - Delta(i) + Delta(k));
            end
        end
    end

    % --- Step 4: Combine Jacobian and solve ---
    J = [J1 J2; J3 J4];
    X = J \ mismatch;   % Faster & stable than inv(J)*mismatch

    % --- Step 5: Update voltages and angles ---
    dDelta = X(1:nBus-1);
    dVmag  = X(nBus:end);

    VmagNew(2:nBus)  = VmagNew(2:nBus)  + dVmag;
    DeltaNew(2:nBus) = DeltaNew(2:nBus) + dDelta;

    disp(J)

    % Update iteration variables
    Vmag  = VmagNew;
    Delta = DeltaNew;
    iteration = iteration + 1;

    % Display iteration 1 values
    if iteration == 1
        fprintf('\nIteration %d\n', iteration);
        fprintf('Bus Voltages (pu):\n');
        disp(Vmag);
        fprintf('Bus Angles (radians):\n');
        disp(Delta);
    end
end

%% ----------------------- OUTPUT -----------------------
fprintf('\nConverged in %d iterations\n', iteration);
fprintf('\nFinal Bus Voltages (pu):\n');
disp(Vmag);
fprintf('Final Bus Angles (radians):\n');
disp(Delta);
