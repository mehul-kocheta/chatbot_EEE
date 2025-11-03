% Filename: calculate_fault.m
%
% Calculates the post-fault bus voltages, total fault current,
% and post-fault bus current injections for a symmetrical 3-phase
% bolted (zero impedance) fault.
%
% Inputs:
%   bus_matrix - The (n x n) complex positive-sequence matrix, can be either:
%                - Zbus (impedance matrix) or
%                - Ybus (admittance matrix)
%   is_zbus    - Boolean flag indicating if bus_matrix is Zbus (true) or Ybus (false)
%   V_pre      - The (n x 1) complex pre-fault bus voltage column vector.
%                (Often, this is just 1.0 pu at all buses).
%   fault_bus  - The 1-based index of the bus where the fault occurs.
%
% Outputs:
%   V_post        - The (n x 1) complex post-fault bus voltage vector.
%   I_fault       - The complex total fault current flowing into the fault.
%   I_post_inject - The (n x 1) complex post-fault current injections.

function [V_post, I_fault, I_post_inject] = calculate_fault(bus_matrix, is_zbus, V_pre, fault_bus)
    % Convert input matrix if needed
    if is_zbus
        Zbus_1 = bus_matrix;
        Ybus_1 = inv(Zbus_1);  % Calculate Ybus as inverse of Zbus
    else
        Ybus_1 = bus_matrix;
        Zbus_1 = inv(Ybus_1);  % Calculate Zbus as inverse of Ybus
    end
    
    % --- 1. Get Pre-Fault Voltage at the Fault Location ---
    % This is the Thevenin voltage (V_th) at the fault bus.
    V_k_pre = V_pre(fault_bus);
    
    % --- 2. Get Thevenin Impedance at the Fault Location ---
    % This is the diagonal element of Zbus corresponding to the fault bus.
    Z_th = Zbus_1(fault_bus, fault_bus);
    
    % --- 3. Calculate Total Fault Current (I_fault) ---
    % Using Thevenin's Law: I_fault = V_th / Z_th
    % (Assuming fault impedance Zf = 0 for a bolted fault)
    I_fault = V_k_pre / Z_th;
    
    % --- 4. Calculate Post-Fault Voltages (V_post) ---
    % The post-fault voltage at any bus 'i' is:
    % V_post(i) = V_pre(i) - Zbus(i, k) * I_fault
    %
    % In matrix form:
    
    % Get the k-th column of Zbus (transfer impedances to bus k)
    Z_k_column = Zbus_1(:, fault_bus);
    
    % Calculate the change in voltage at all buses due to the fault
    delta_V = Z_k_column * I_fault;
    
    % Subtract the voltage change from the pre-fault voltages
    V_post = V_pre - delta_V;
    
    % --- 5. Calculate Post-Fault Current Injections ---
    % Using the post-fault voltages and Ybus: I_inject = Ybus * V_post
    % This vector shows the currents being injected at each bus.
    % - At generator buses, it's the generator's fault contribution.
    % - At the fault bus (k), I_post_inject(k) should equal -I_fault.
    I_post_inject = Ybus_1 * V_post;
    
end