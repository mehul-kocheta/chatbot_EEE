import os
import json
import base64
import re
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from groq import Groq
from dotenv import load_dotenv
import sys
import textwrap

# Add agents directory to sys.path
agents_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents")
sys.path.append(agents_dir)

# Import the agent directly (skipping agents/__init__.py to avoid dependency errors)
try:
    import matlab_executor_agent
    run_matlab_executor_agent = matlab_executor_agent.run_matlab_executor_agent
except ImportError:
    # Fallback if structure is different
    from agents.matlab_executor_agent import run_matlab_executor_agent

load_dotenv()

client = Groq()
# MODEL = "openai/gpt-oss-120b" # As seen in orchestrator
MODEL = "llama-3.3-70b-versatile" # Using a standard Groq model for reliability

def generate_questions():
    print("Requesting 10 MATLAB plotting questions from Groq...")
    system_msg = (
        "You are an expert in MATLAB and Control Systems. "
        "Generate 10 diverse technical questions that involve plotting in MATLAB. "
        "Topics should include transform equations (Laplace, Z-transform), response analysis (step, impulse), "
        "bode plots, and other type of responses like diode or PV array"
        "Provide clear, standalone questions that can be executed as individual tasks. "
        "Return the questions as a JSON object with a key 'questions' containing a list of strings."
    )
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": "Generate 10 MATLAB plotting questions for engineering students."}
            ],
            response_format={"type": "json_object"}
        )
        
        data = json.loads(response.choices[0].message.content)
        questions = data.get("questions", [])
        print(f"Generated {len(questions)} questions.")
        return questions
    except Exception as e:
        print(f"Error generating questions: {e}")
        return []

def extract_base64_plots(response_text):
    # Pattern to match ![plot](data:image/png;base64,<data>)
    pattern = r"!\[plot\]\(data:image/png;base64,([A-Za-z0-9+/=]+)\)"
    return re.findall(pattern, response_text)

def create_grid(results, output_path="matlab_plots_grid.png"):
    if not results:
        print("No results to plot.")
        return

    print(f"Creating professional grid for {len(results)} plots...")
    
    num_plots = len(results)
    cols = 2
    rows = (num_plots + cols - 1) // cols
    
    # Increase figure size for high quality
    fig, axes = plt.subplots(rows, cols, figsize=(15, 6 * rows), constrained_layout=True)
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Configure global styling for research paper
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 300
    })

    for i, (question, b64_data) in enumerate(results):
        ax = axes[i]
        
        try:
            # Decode image
            img_data = base64.b64decode(b64_data)
            img = Image.open(BytesIO(img_data))
            
            # Display image
            ax.imshow(img)
            ax.axis('off') # Hide axes for a clean look
            
            # Add subfigure label (a, b, c...)
            label = chr(97 + i) # 'a', 'b', 'c', ...
            # Wrapping question text for title
            wrapped_title = textwrap.fill(f"({label}) {question}", width=60)
            ax.set_title(wrapped_title, pad=15, fontweight='bold')
            
        except Exception as e:
            print(f"Error processing image for Q{i+1}: {e}")
            ax.text(0.5, 0.5, f"Error Loading Image\nQ{i+1}", ha='center', va='center')
            ax.axis('off')

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Professional grid successfully saved to {output_path}")

def main():
    questions = generate_questions()
    if not questions:
        print("No questions generated. Using fallbacks.")
        questions = [
            "Plot the step response of G(s) = 5/(s^2 + 3s + 2)",
            "Plot the bode response of H(s) = 1/(s+10)",
            "Generate a root locus for K / (s(s+1)(s+2))",
            "Plot the impulse response of Y(s) = s / (s^2 + 4)",
            "Visualize the step response of a mass-spring-damper system: m=1, c=0.5, k=2",
            "Plot the Z-transform response of H(z) = 1 / (1 - 0.5z^-1)",
            "Compare the step response of open-loop and closed-loop system for G(s) = 1/s",
            "Plot the frequency response of a low-pass filter with cutoff at 100 rad/s",
            "Show the state-space response for dx/dt = [0 1; -2 -3]x + [0; 1]u",
            "Plot the response of a second-order system with omega_n=5 and damping ratio=0.7"
        ]
    
    results = []
    for i, q in enumerate(questions):
        # Use a safe way to print potential non-ASCII characters on Windows terminals
        safe_q = q.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)
        print(f"\n--- Processing Q{i+1}/10 ---")
        print(f"Question: {safe_q}")
        try:
            # Call the agent
            response = run_matlab_executor_agent(q)
            plots = extract_base64_plots(response)
            
            if plots:
                results.append((q, plots[0]))
                print(f"Success: Plot captured for Q{i+1}")
            else:
                print(f"Warning: No plot found in agent response for Q{i+1}")
                # Some agents might fail or produce pure text, we try to append at least the question
                # or just skip if we want a clean grid.
        except Exception as e:
            error_msg = str(e).encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)
            print(f"Error calling agent for Q{i+1}: {error_msg}")
            
    if results:
        create_grid(results)
    else:
        print("No results collected. Grid creation aborted.")

if __name__ == "__main__":
    main()
