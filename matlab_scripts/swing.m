clc;
clear all;

Pa = 0.9;
Pm = 0.9;
E = 1.1;
V = 1;
Xdf = 1.25;
Xaf = 0.45;
Pm_max = (E*V)/Xaf;
delta = [21.64];
T = 0.45;
tc = 0.5;
t = 0;
time = [0];
m = 2.52/(180*50);
ddelta = 0;

while t < T
    if t > tc
        X = Xaf;
    else
        X = Xdf;
    end
    delta_rad = delta(end)*(pi/180);
    Pm_max = (E * V) / X;
    Pa = Pm - Pm_max*sin(delta_rad);

    ddelta = ddelta + ((0.05)^2)*(Pa/m);
    delta_rad = (delta_rad*(180/pi) + ddelta)*(pi/180);

    delta_deg = delta_rad*(180/pi);

    t = t + 0.05;
    delta = [delta delta_deg];
    time = [time t];
end

plot(time, delta);

