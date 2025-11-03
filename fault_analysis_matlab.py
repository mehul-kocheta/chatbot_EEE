import matlab.engine
import numpy as np

def get_fault_analysis_matlab(bus_matrix_np, is_zbus, v_pre_np, fault_bus_py):
    """
    Calculate post-fault conditions using MATLAB engine.
    
    Args:
        bus_matrix_np: Complex numpy array of either Zbus or Ybus matrix
        is_zbus: Boolean indicating if bus_matrix is Zbus (True) or Ybus (False)
        v_pre_np: Complex numpy array of pre-fault voltages
        fault_bus_py: Zero-based index of fault bus location
        
    Returns:
        Tuple of (v_post, i_fault, i_post_inject) as numpy arrays
    """
    # Start MATLAB engine
    eng = matlab.engine.start_matlab()
    
    # Convert numpy arrays to MATLAB arrays
    bus_matrix_m = matlab.double(
        [[x for x in row] for row in bus_matrix_np.tolist()],
        is_complex=True
    )

    v_pre_m = matlab.double(
        [[x] for x in v_pre_np.tolist()],
        is_complex=True
    )
    
    # Convert 0-based Python index to 1-based MATLAB index
    fault_bus_m = fault_bus_py + 1
    
    try:
        # Call MATLAB function
        v_post_m, i_fault_m, i_post_inject_m = eng.calculate_fault(
            bus_matrix_m, is_zbus, v_pre_m, fault_bus_m, nargout=3
        )
        
        # Convert MATLAB outputs back to numpy arrays
        v_post_np = np.array([complex(x[0].real, x[0].imag) for x in v_post_m])
        i_fault = complex(i_fault_m.real, i_fault_m.imag)
        i_post_inject_np = np.array([complex(x[0].real, x[0].imag) for x in i_post_inject_m])
        
        return v_post_np, i_fault, i_post_inject_np
        
    finally:
        # Always quit MATLAB engine
        eng.quit()