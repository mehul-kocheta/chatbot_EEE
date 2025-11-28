import numpy as np
import json
from groq import Groq
from dotenv import load_dotenv
import os
import matlab.engine

load_dotenv()

client = Groq()
MODEL = "openai/gpt-oss-120b"

def compute_ybus_matlab(line_data):
    """
    Compute Ybus matrix using MATLAB engine
    
    Args:
        line_data: List of lists where each row is [from_bus, to_bus, R, X, a, shunt]
    
    Returns:
        Ybus matrix as a numpy array
    """
    # Start MATLAB engine
    eng = matlab.engine.start_matlab()
    
    # Convert line_data to MATLAB format
    ldata = matlab.double(line_data)
    
    # Extract columns
    fb = eng.eval(f"ldata(:,1)", nargout=1)
    tb = eng.eval(f"ldata(:,2)", nargout=1)
    R = eng.eval(f"ldata(:,3)", nargout=1)
    X = eng.eval(f"ldata(:,4)", nargout=1)
    a = eng.eval(f"ldata(:,5)", nargout=1)
    sh = eng.eval(f"ldata(:,6)", nargout=1)
    
    # Store in MATLAB workspace
    eng.workspace['ldata'] = ldata
    
    # Execute MATLAB code to compute Ybus
    eng.eval("fb = ldata(:,1);", nargout=0)
    eng.eval("tb = ldata(:,2);", nargout=0)
    eng.eval("R = ldata(:,3);", nargout=0)
    eng.eval("X = ldata(:,4);", nargout=0)
    eng.eval("a = ldata(:,5);", nargout=0)
    eng.eval("sh = ldata(:,6);", nargout=0)
    eng.eval("z = R + 1i*X;", nargout=0)
    eng.eval("nbus = max(max(fb), max(tb));", nargout=0)
    eng.eval("nbranch = length(fb);", nargout=0)
    eng.eval("ybus = zeros(nbus, nbus);", nargout=0)
    
    # Build off-diagonal elements
    eng.eval("""
    for m = 1:nbranch
        ybus(fb(m), tb(m)) = -1/(z(m)*a(m));
        ybus(tb(m), fb(m)) = -1/(z(m)*a(m));
    end
    """, nargout=0)
    
    # Build diagonal elements
    eng.eval("""
    for m = 1:nbranch
        ybus(fb(m), fb(m)) = ybus(fb(m), fb(m)) + 1/(z(m)*(a(m)^2)) + sh(m)/2;
        ybus(tb(m), tb(m)) = ybus(tb(m), tb(m)) + 1/(z(m)*(a(m)^2)) + sh(m)/2;
    end
    """, nargout=0)
    
    # Get the result
    ybus = eng.workspace['ybus']
    
    # Stop MATLAB engine
    eng.quit()
    
    # Convert to numpy array
    ybus_array = np.array(ybus)
    
    return ybus_array

def run_ybus_agent(user_prompt):
    """
    Agent that computes Ybus matrix from branch data using LLM + MATLAB
    """
    messages = [
        {
            "role": "system",
            "content": """You are a power system Ybus calculation expert. 
            Your job is to:
            1. Parse the user's input to extract branch/line data
            2. Branch data format: From Bus, To Bus, R (resistance), X (reactance), a (transformer ratio), Shunt Admittance
            3. Call the compute_ybus function with the extracted data
            4. Return the Ybus matrix in a clear format
            
            The line data should be in the format:
            [[from1, to1, R1, X1, a1, sh1],
             [from2, to2, R2, X2, a2, sh2],
             ...]
            
            If transformer ratio or shunt admittance is not provided, use 1 for transformer ratio and 0 for shunt.
            
            Output your answer in pure markdown format with the Ybus matrix clearly displayed."""
        },
        {
            "role": "user",
            "content": user_prompt,
        }
    ]
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "compute_ybus",
                "description": "Compute Ybus matrix from branch/line data using MATLAB",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "line_data": {
                            "type": "array",
                            "description": "Array of branch data where each row is [from_bus, to_bus, R, X, a, shunt]",
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "number"
                                }
                            }
                        }
                    },
                    "required": ["line_data"]
                }
            }
        }
    ]
    
    available_functions = {
        "compute_ybus": lambda line_data: str(compute_ybus_matlab(line_data))
    }
    
    max_iterations = 5
    iteration = 0
    
    while iteration < max_iterations:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=False,
            tools=tools,
            tool_choice="auto",
            max_tokens=8100
        )
        
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        
        messages.append(response_message)
        
        if not tool_calls:
            return response_message.content
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            
            # Call the function
            if function_name == "compute_ybus":
                function_response = function_to_call(function_args["line_data"])
            else:
                function_response = function_to_call(**function_args)
            
            print(f"{function_name} result: {function_response}")
            
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )
        
        iteration += 1
    
    return response_message.content if response_message.content else "Maximum iterations reached."

if __name__ == "__main__":
    # Test with the example data from the MATLAB code
    test_query = """
    Calculate Ybus for the following system:
    Branch 1: From bus 1 to bus 2, R=0.03, X=0.08, transformer ratio=1, shunt=0.04
    Branch 2: From bus 1 to bus 3, R=0.02, X=0.05, transformer ratio=1, shunt=0.02
    Branch 3: From bus 2 to bus 3, R=0.01, X=0.03, transformer ratio=1, shunt=0.03
    """
    result = run_ybus_agent(test_query)
    print(result)

