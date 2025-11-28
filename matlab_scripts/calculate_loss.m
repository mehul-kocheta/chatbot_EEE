% Filename: calculate_loss.m
%
% Calculates the total real power loss in a system given the 
% system admittance matrix (Ybus) and the final bus voltage vector (V).
%
% Inputs:
%   Ybus - The (n x n) complex system admittance matrix.
%   V    - The (n x 1) complex bus voltage column vector.
%
% Output:
%   Ploss - The total real power loss (a scalar value).

function Ploss = calculate_loss(Ybus, V)
    % Calculate the vector of complex current injections
    I = Ybus * V;
    
    % Calculate the vector of complex power injections (S = V .* conj(I))
    % S_inject_k = P_gen_k - P_load_k + j*(Q_gen_k - Q_load_k)
    S_inject = V .* conj(I);
    
    % The total system loss is the sum of all complex power injections
    % S_loss = sum(S_gen) - sum(S_load) = sum(S_gen - S_load)
    S_loss_total = sum(S_inject);
    
    % Extract the real part for the total real power loss
    Ploss = real(S_loss_total);
end
