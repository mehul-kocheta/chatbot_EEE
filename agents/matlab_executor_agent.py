import matlab.engine
import numpy as np
import json
import re
import os
from groq import Groq
from dotenv import load_dotenv
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

load_dotenv()

client = Groq()
MODEL = "openai/gpt-oss-120b"

def extract_matlab_code(text):
    """
    Extract MATLAB code from markdown code blocks
    """
    # Try to find code in ```matlab or ```MATLAB blocks
    matlab_pattern = r"```(?:matlab|MATLAB)\n(.*?)```"
    matches = re.findall(matlab_pattern, text, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # Try generic code blocks
    generic_pattern = r"```\n(.*?)```"
    matches = re.findall(generic_pattern, text, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    return None

def execute_matlab_calculation(matlab_code):
    """
    Execute MATLAB code for calculations (non-plotting) and return text output
    
    Returns:
        dict with 'output' (text), 'error' (if any)
    """
    result = {
        'output': '',
        'error': None
    }
    
    try:
        print("Starting MATLAB engine for calculations...")
        eng = matlab.engine.start_matlab()
        
        # Capture output
        from io import StringIO
        output_buffer = StringIO()
        
        print("Executing MATLAB code...")
        try:
            eng.eval(matlab_code, nargout=0, stdout=output_buffer)
            result['output'] = output_buffer.getvalue()
        except Exception as eval_error:
            result['error'] = str(eval_error)
            print(f"MATLAB execution error: {eval_error}")
        
        print("Stopping MATLAB engine...")
        eng.quit()
        
    except Exception as e:
        result['error'] = f"Failed to execute MATLAB code: {str(e)}"
        print(result['error'])
    
    return result

def execute_matlab_for_plot_data(matlab_code):
    """
    Execute MATLAB code to extract plot data (x, y), then use matplotlib for plotting
    
    The MATLAB code should store plot data in variables like:
    - x_data, y_data (or x1, y1, x2, y2 for multiple plots)
    - plot_title, plot_xlabel, plot_ylabel, plot_legends (optional metadata)
    
    Returns:
        dict with 'output' (text), 'plots' (list of base64 encoded images), 'error' (if any)
    """
    result = {
        'output': '',
        'plots': [],
        'error': None
    }
    
    try:
        print("Starting MATLAB engine for plot data extraction...")
        eng = matlab.engine.start_matlab()
        
        # Capture output
        from io import StringIO
        output_buffer = StringIO()
        
        print("Executing MATLAB code to compute plot data...")
        try:
            eng.eval(matlab_code, nargout=0, stdout=output_buffer)
            result['output'] = output_buffer.getvalue()
            
            # Get workspace variables
            workspace_vars = eng.eval("who", nargout=1)
            print(f"Workspace variables: {workspace_vars}")
            
            # Extract plot metadata if available
            plot_title = "Plot"
            plot_xlabel = "X"
            plot_ylabel = "Y"
            plot_legends = []
            
            if 'plot_title' in workspace_vars:
                plot_title = str(eng.workspace['plot_title'])
            if 'plot_xlabel' in workspace_vars:
                plot_xlabel = str(eng.workspace['plot_xlabel'])
            if 'plot_ylabel' in workspace_vars:
                plot_ylabel = str(eng.workspace['plot_ylabel'])
            if 'plot_legends' in workspace_vars:
                legends_cell = eng.workspace['plot_legends']
                plot_legends = [str(leg) for leg in legends_cell]
            
            # Extract plot data - look for x_data, y_data or x1, y1, x2, y2, etc.
            plot_data_sets = []
            
            # Try single plot data first
            if 'x_data' in workspace_vars and 'y_data' in workspace_vars:
                x = np.array(eng.workspace['x_data']).flatten()
                y = np.array(eng.workspace['y_data']).flatten()
                plot_data_sets.append({'x': x, 'y': y, 'label': 'Data'})
            
            # Try numbered plot data (x1, y1, x2, y2, etc.)
            else:
                idx = 1
                while f'x{idx}' in workspace_vars and f'y{idx}' in workspace_vars:
                    x = np.array(eng.workspace[f'x{idx}']).flatten()
                    y = np.array(eng.workspace[f'y{idx}']).flatten()
                    label = plot_legends[idx-1] if idx-1 < len(plot_legends) else f'Plot {idx}'
                    plot_data_sets.append({'x': x, 'y': y, 'label': label})
                    idx += 1
            
            # If no standard variables found, try common patterns (t, y), (x, y), etc.
            if not plot_data_sets:
                common_x = ['t', 'time', 'x', 'freq', 'w']
                common_y = ['y', 'output', 'response', 'mag', 'magnitude']
                
                x_var = None
                y_vars = []
                
                for var in common_x:
                    if var in workspace_vars:
                        x_var = var
                        break
                
                for var in common_y:
                    if var in workspace_vars:
                        y_vars.append(var)
                
                if x_var and y_vars:
                    x = np.array(eng.workspace[x_var]).flatten()
                    for y_var in y_vars:
                        y = np.array(eng.workspace[y_var]).flatten()
                        plot_data_sets.append({'x': x, 'y': y, 'label': y_var})
            
            # Create matplotlib plot
            if plot_data_sets:
                plt.figure(figsize=(10, 6))
                
                for dataset in plot_data_sets:
                    plt.plot(dataset['x'], dataset['y'], label=dataset['label'], linewidth=2)
                
                plt.xlabel(plot_xlabel, fontsize=12)
                plt.ylabel(plot_ylabel, fontsize=12)
                plt.title(plot_title, fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3)
                
                if len(plot_data_sets) > 1 or plot_legends:
                    plt.legend()
                
                # Convert to base64
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                result['plots'].append(plot_base64)
                plt.close()
                
                print(f"Successfully created plot with {len(plot_data_sets)} dataset(s)")
            else:
                result['error'] = "Could not find plot data in MATLAB workspace. Please ensure data is stored in variables like x_data, y_data or x1, y1, x2, y2, etc."
            
        except Exception as eval_error:
            result['error'] = str(eval_error)
            print(f"MATLAB execution error: {eval_error}")
        
        print("Stopping MATLAB engine...")
        eng.quit()
        
    except Exception as e:
        result['error'] = f"Failed to execute MATLAB code: {str(e)}"
        print(result['error'])
    
    return result

def run_matlab_executor_agent(user_prompt):
    """
    Agent that generates and executes MATLAB code based on user query
    Uses tool calling to differentiate between plotting and calculation tasks
    """
    messages = [
        {
            "role": "system",
            "content": """You are a MATLAB code generation expert. 
            
            Your job is to:
            1. Analyze the user's request to determine if it requires PLOTTING or CALCULATION
            2. Generate appropriate MATLAB code
            3. Call the execute_matlab_code tool with the code and is_plot_code flag
            
            CRITICAL INSTRUCTIONS:
            
            FOR PLOTTING TASKS (is_plot_code = true):
            - Generate MATLAB code that COMPUTES and STORES plot data
            - DO NOT use plot(), figure(), or any plotting commands
            - Store x-axis data in: x_data (or x1, x2, x3 for multiple plots)
            - Store y-axis data in: y_data (or y1, y2, y3 for multiple plots)
            - Optionally store metadata:
              * plot_title = 'Your Title';
              * plot_xlabel = 'X Label';
              * plot_ylabel = 'Y Label';
              * plot_legends = {'Legend 1', 'Legend 2'};
            
            Example for step response (PLOTTING):
            ```matlab
            % Define system
            num = [5];
            den = [1, 3, 2];
            sys = tf(num, den);
            
            % Compute step response data (NO PLOTTING)
            [y_data, x_data] = step(sys);
            
            % Store metadata for plotting
            plot_title = 'Step Response';
            plot_xlabel = 'Time (seconds)';
            plot_ylabel = 'Amplitude';
            ```
            
            FOR CALCULATION TASKS (is_plot_code = false):
            - Generate normal MATLAB code with calculations
            - Use disp() or fprintf() to display results
            - Can perform any mathematical operations
            
            Example for calculations:
            ```matlab
            A = [1, 2, 3; 4, 5, 6; 7, 8, 9];
            det_A = det(A);
            eigenvalues = eig(A);
            
            fprintf('Determinant: %.4f\\n', det_A);
            disp('Eigenvalues:');
            disp(eigenvalues);
            ```
            
            Always call the execute_matlab_code tool with:
            - matlab_code: Your generated code
            - is_plot_code: true for plotting, false for calculations"""
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
                "name": "execute_matlab_code",
                "description": "Execute MATLAB code. Use is_plot_code=true for plotting tasks (data extraction + matplotlib), is_plot_code=false for calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "matlab_code": {
                            "type": "string",
                            "description": "The MATLAB code to execute"
                        },
                        "is_plot_code": {
                            "type": "boolean",
                            "description": "True if this is for plotting (will extract data and use matplotlib), False for calculations"
                        }
                    },
                    "required": ["matlab_code", "is_plot_code"]
                }
            }
        }
    ]
    
    max_iterations = 5
    iteration = 0
    last_execution_result = None  # Store the last execution result
    last_matlab_code = None  # Store the last MATLAB code
    
    while iteration < max_iterations:
        print(f"\nIteration {iteration + 1}/{max_iterations}")
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            stream=False,
            max_tokens=4000
        )
        
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        
        messages.append(response_message)
        
        if not tool_calls:
            # No tool call, return the message content
            if last_execution_result and not last_execution_result.get('error'):
                # Format final response with execution results
                return format_final_response(response_message.content, last_matlab_code, last_execution_result)
            return response_message.content if response_message.content else "I couldn't generate a solution."
        
        # Process tool calls
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"\nTool call: {function_name}")
            print(f"Arguments: is_plot_code={function_args.get('is_plot_code')}")
            
            if function_name == "execute_matlab_code":
                matlab_code = function_args["matlab_code"]
                is_plot_code = function_args.get("is_plot_code", False)
                
                last_matlab_code = matlab_code
                
                print(f"\nExecuting MATLAB code (is_plot_code={is_plot_code})...")
                print(f"Code preview:\n{matlab_code[:200]}...\n")
                
                # Execute based on type
                if is_plot_code:
                    execution_result = execute_matlab_for_plot_data(matlab_code)
                else:
                    execution_result = execute_matlab_calculation(matlab_code)
                
                last_execution_result = execution_result
                
                # Build response string for the tool
                response_parts = []
                
                if execution_result.get('error'):
                    response_parts.append(f"Error: {execution_result['error']}")
                else:
                    response_parts.append("Execution successful!")
                    if execution_result.get('output'):
                        response_parts.append(f"Output:\n{execution_result['output']}")
                    
                    if execution_result.get('plots'):
                        response_parts.append(f"\nGenerated {len(execution_result['plots'])} plot(s) successfully.")
                
                function_response = "\n".join(response_parts)
                
                # Add tool response to messages
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response
                })
        
        iteration += 1
    
    # Get final response with formatting
    final_response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=False,
        max_tokens=2000
    )
    
    final_content = final_response.choices[0].message.content
    
    # Format and return with execution results
    if last_execution_result:
        return format_final_response(final_content, last_matlab_code, last_execution_result)
    
    return final_content

def format_final_response(llm_response, matlab_code, execution_result):
    """
    Format the final response with MATLAB code, output, and plots
    """
    response_parts = []
    
    # Add LLM explanation if available
    if llm_response:
        response_parts.append(llm_response)
    
    # Add MATLAB code
    if matlab_code:
        response_parts.append(f"\n\n**MATLAB Code:**\n```matlab\n{matlab_code}\n```")
    
    # Add execution output
    if execution_result.get('output'):
        response_parts.append(f"\n\n**Execution Output:**\n```\n{execution_result['output']}\n```")
    
    # Add plots
    if execution_result.get('plots'):
        response_parts.append(f"\n\n**Generated Plot(s):**")
        for idx, plot_data in enumerate(execution_result['plots'], 1):
            response_parts.append(f"\n![Plot {idx}](data:image/png;base64,{plot_data})")
    
    # Add error if present
    if execution_result.get('error'):
        response_parts.append(f"\n\n**Error:**\n```\n{execution_result['error']}\n```")
    
    return "\n".join(response_parts)

if __name__ == "__main__":
    # Test 1: Plotting query
    print("\n" + "="*80)
    print("TEST 1: PLOTTING TASK")
    print("="*80)
    test_query_plot = """
    Plot the step response of the transfer function H(s) = 5 / (s^2 + 3s + 2)
    """
    result1 = run_matlab_executor_agent(test_query_plot)
    print("\nFINAL RESULT:")
    print(result1)
    
    print("\n\n" + "="*80)
    print("TEST 2: CALCULATION TASK")
    print("="*80)
    # Test 2: Calculation query
    test_query_calc = """
    Create a 3x3 matrix with values [[1,2,3],[4,5,6],[7,8,9]] and calculate its determinant and eigenvalues. Display the results.
    """
    result2 = run_matlab_executor_agent(test_query_calc)
    print("\nFINAL RESULT:")
    print(result2)

