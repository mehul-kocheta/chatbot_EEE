# Ybus Calculation Agent

## Overview

The Ybus agent is a new addition to the power system analysis toolkit that computes the bus admittance matrix (Ybus) from branch/line data using MATLAB engine integration.

## Features

- **Automated Ybus Calculation**: Converts branch data (resistance, reactance, transformer ratios, shunt admittance) into a complete bus admittance matrix
- **MATLAB Integration**: Uses `matlab.engine` for accurate calculations following standard power system formulas
- **LLM-Powered Parsing**: Intelligently extracts branch data from natural language inputs
- **Seamless Integration**: Works as part of the orchestrated power flow agent system

## Prerequisites

1. **MATLAB Installation**: You need MATLAB installed on your system
2. **MATLAB Engine for Python**: Install using:
   ```bash
   cd "matlabroot\extern\engines\python"
   python setup.py install
   ```
   Replace `matlabroot` with your actual MATLAB installation path

3. **Python Dependencies**: Ensure you have the required packages:
   ```bash
   pip install numpy groq python-dotenv
   ```

## Branch Data Format

Each branch in the power system requires the following parameters:

| Parameter | Description | Unit |
|-----------|-------------|------|
| From Bus | Starting bus number | - |
| To Bus | Ending bus number | - |
| R | Resistance | per unit (pu) |
| X | Reactance | per unit (pu) |
| a | Transformer ratio | - |
| Shunt | Shunt admittance | per unit (pu) |

**Default Values:**
- If transformer ratio (`a`) is not specified, it defaults to 1
- If shunt admittance is not specified, it defaults to 0

## Usage

### Standalone Usage

```python
from agents.ybus_agent import run_ybus_agent

query = """
Calculate Ybus for the following system:
Branch 1: From bus 1 to bus 2, R=0.03, X=0.08, transformer ratio=1, shunt=0.04
Branch 2: From bus 1 to bus 3, R=0.02, X=0.05, transformer ratio=1, shunt=0.02
Branch 3: From bus 2 to bus 3, R=0.01, X=0.03, transformer ratio=1, shunt=0.03
"""

result = run_ybus_agent(query)
print(result)
```

### Integrated with Power Flow Agent

The Ybus agent is automatically available when using the main power flow agent:

```python
from agents.power_flow_agent import run_power_flow_agent

query = """
I have a 3-bus system with branch data:
- Branch 1-2: R=0.03, X=0.08, a=1, shunt=0.04
- Branch 1-3: R=0.02, X=0.05, a=1, shunt=0.02
- Branch 2-3: R=0.01, X=0.03, a=1, shunt=0.03

Calculate the Ybus matrix and then solve power flow with:
- Bus 1 as slack (V=1.0∠0°)
- Bus 2 load: P=0.5 pu
- Bus 3 load: P=0.3 pu
"""

result = run_power_flow_agent(query)
print(result)
```

## How It Works

1. **Input Parsing**: The LLM extracts branch data from natural language input
2. **MATLAB Execution**: The data is converted to MATLAB format and the Ybus calculation is performed using standard formulas:
   - Off-diagonal elements: `Y[i,j] = -1/(Z * a)` where Z = R + jX
   - Diagonal elements: `Y[i,i] = sum of admittances of all branches connected to bus i + shunt/2`
3. **Result Formatting**: The Ybus matrix is returned in a readable format

## MATLAB Code Reference

The agent implements the following MATLAB algorithm:

```matlab
% Off-diagonal elements
for m = 1:nbranch
    ybus(fb(m), tb(m)) = -1/(z(m)*a(m));
    ybus(tb(m), fb(m)) = -1/(z(m)*a(m));
end

% Diagonal elements
for m = 1:nbranch
    ybus(fb(m), fb(m)) = ybus(fb(m), fb(m)) + 1/(z(m)*(a(m)^2)) + sh(m)/2;
    ybus(tb(m), tb(m)) = ybus(tb(m), tb(m)) + 1/(z(m)*(a(m)^2)) + sh(m)/2;
end
```

## Integration with Other Agents

The Ybus agent works seamlessly with other agents in the system:

1. **Ybus Agent** → Calculates bus admittance matrix
2. **Power Flow Agent (GS)** → Uses Ybus to solve for bus voltages
3. **Loss Agent** → Calculates system losses using Ybus and voltages
4. **Fault Agent** → Analyzes fault conditions using Ybus/Zbus

## Example Test

Run the test file to see the agent in action:

```bash
python test_ybus.py
```

This will demonstrate:
- Simple Ybus calculation
- Integrated workflow (Ybus → Power Flow)

## Troubleshooting

### MATLAB Engine Not Found
If you get an error about MATLAB engine not being found:
1. Ensure MATLAB is installed
2. Install MATLAB Engine for Python (see Prerequisites)
3. Verify installation: `python -c "import matlab.engine"`

### Incorrect Ybus Results
- Verify branch data format (especially complex numbers for impedance)
- Check that bus numbering is consecutive (1, 2, 3, ...)
- Ensure per-unit values are correct

## API Reference

### `run_ybus_agent(user_prompt: str) -> str`

Main entry point for the Ybus calculation agent.

**Parameters:**
- `user_prompt` (str): Natural language description of the power system branch data

**Returns:**
- `str`: Markdown-formatted result containing the Ybus matrix

### `compute_ybus_matlab(line_data: list) -> np.ndarray`

Core function that performs Ybus calculation using MATLAB.

**Parameters:**
- `line_data` (list): 2D list where each row is `[from_bus, to_bus, R, X, a, shunt]`

**Returns:**
- `np.ndarray`: Complex-valued Ybus matrix

## License

Part of the Major Project power system analysis toolkit.

