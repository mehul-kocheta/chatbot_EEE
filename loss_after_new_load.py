import matlab.engine
import numpy as np
import os

def get_total_loss_matlab(ybus_np, v_np, new_load, bus_at_py):
    """
    Uses the MATLAB Engine to calculate total system power loss.
    
    NOTE ON ASSUMPTIONS:
    This function calculates the total system loss (Ploss) using the
    formula: Ploss = real(sum(V .* conj(Ybus * V))). 
    
    This formula requires *only* the Ybus and the *final* steady-state
    voltage vector (V). 
    
    Therefore, it is ASSUMED that the input `v_np` is the voltage profile
    *after* the `new_load` has been added and the power flow has been
    re-solved. 
    
    The inputs `new_load` and `bus_at_py` are not used in the 
    'calculate_loss.m' script itself, as their effects are
    already captured in the `v_np` vector. They are included here
    to match your requested function signature.

    Args:
        ybus_np (np.ndarray): The (n, n) complex Y-bus matrix.
        v_np (np.ndarray): The (n,) or (n, 1) complex voltage vector.
        new_load (complex): The new load value (e.g., 1.0 + 0.5j for 1pu P, 0.5pu Q).
        bus_at_py (int): The 0-based Python index of the bus.

    Returns:
        float: The total real power loss, or None if an error occurs.
    """
    
    print("Starting MATLAB engine...")
    try:
        eng = matlab.engine.start_matlab()
        print("MATLAB engine started.")
        
        # Add the current directory to the MATLAB path to find the .m file
        current_dir = os.path.dirname(os.path.realpath(__file__))
        eng.addpath(current_dir, nargout=0)
        print(f"Added '{current_dir}' to MATLAB path.")

        # --- Data Conversion ---
        # 1. Convert Ybus to MATLAB complex double
        ybus_m = matlab.double(ybus_np.tolist(), is_complex=True)
        
        # 2. Ensure V is a column vector (n, 1) for MATLAB
        if v_np.ndim == 1:
            v_col_np = v_np.reshape(-1, 1)
        else:
            v_col_np = v_np
            
        # 3. Convert V to MATLAB complex double
        v_m = matlab.double(v_col_np.tolist(), is_complex=True)
        
        print("Data converted for MATLAB.")

        # --- Call MATLAB Function ---
        # The 'calculate_loss.m' script must be in the same directory
        # or on the MATLAB path.
        print("Calling 'calculate_loss' function in MATLAB...")
        ploss = eng.calculate_loss(ybus_m, v_m)
        print(f"MATLAB calculation complete. Total Loss: {ploss}")

        # --- Cleanup ---
        eng.quit()
        print("MATLAB engine stopped.")
        
        return ploss

    except Exception as e:
        print(f"An error occurred with the MATLAB engine: {e}")
        if 'eng' in locals() and eng:
            eng.quit()
        return None

# --- Example Usage ---
# if __name__ == "__main__":
#     # Create a simple 2-bus system example
#     # Line impedance z = 0.05 + 0.1j
#     # Line admittance y = 1/z = 4.0 - 8.0j
#     y_line = 4.0 - 8.0j
    
#     # Ybus matrix (ignoring shunts for simplicity)
#     # Ybus = [[y_line, -y_line], [-y_line, y_line]]
#     ybus = np.array([
#         [y_line, -y_line],
#         [-y_line, y_line]
#     ], dtype=complex)
    
#     # Assume a final voltage profile (e.g., from a solved power flow)
#     # Bus 1 (Slack): V1 = 1.0 + 0.0j
#     # Bus 2 (Load):  V2 = 0.95 - 0.0828j (approx 0.953 < -5 deg)
#     v_final = np.array([1.0 + 0.0j, 0.9464 - 0.0828j], dtype=complex)

#     # Parameters from the prompt (not used in calculation, but part of signature)
#     load = 1.0 + 0.5j # 1.0 PU real power, 0.5 PU reactive
#     bus_index = 1     # Python index for Bus 2
    
#     print("--- Running Power Loss Calculation ---")
    
#     # Check if calculate_loss.m exists
#     if not os.path.exists('calculate_loss.m'):
#         print("\nERROR: 'calculate_loss.m' not found.")
#         print("Please create the 'calculate_loss.m' file in the same directory.")
#     else:
#         total_loss = get_total_loss_matlab(ybus, v_final, load, bus_index)
        
#         if total_loss is not None:
#             print(f"\n--- Final Result ---")
#             print(f"Ybus:\n{ybus}")
#             print(f"Voltages:\n{v_final}")
#             print(f"Total Real Power Loss: {total_loss:.6f} PU")
            
#             # Manual verification (as calculated in the thought process)
#             # I = Ybus @ V = [0.8768-0.0974j, -0.8768+0.0974j]
#             # S_inj = V * conj(I) = [0.8768+0.0974j, -0.8381-0.0196j]
#             # S_loss = sum(S_inj) = 0.0387 + 0.0778j
#             # Ploss = 0.0387
#             print(f"(Expected loss for this V/Ybus is ~0.0387)")
