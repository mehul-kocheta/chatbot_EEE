clc;
clear all;

alpha = [0.02 0.04];
beta = [16 20];

lambda = 50;
lambda_prev = lambda;

Pg_min = 0;
Pg_max = 250;
B = [0.0010 0
    0 0];

Pg = 100*ones(2,1);
load = 237.4;
tol = 1;
sigma = zeros(2,1);
P_loss = 0;

while abs(sum(Pg) - load- P_loss) > tol
    for i=1:2
        sigma(i) = B(i,:)*Pg - B(i,i)*Pg(i);
        Pg(i) = (1-beta(i)/(lambda - 2*sigma(i)))/(alpha(i)/lambda + 2*B(i,i));
    end

    for i=1:2
        if Pg(i) > Pg_max
            Pg(i) = Pg_max;
        elseif Pg(i) < Pg_min
            Pg(i) = Pg_min;
        end
    end

    P_loss = Pg' * B * Pg;

    mismatch = sum(Pg) - load - P_loss;

    if mismatch > 0
        lambda_prev = lambda;
        lambda = lambda - 0.25;
    else
        lambda_prev = lambda;
        lambda = lambda + 0.25;
    end
    disp(Pg)
end

disp(Pg)
disp(P_loss)