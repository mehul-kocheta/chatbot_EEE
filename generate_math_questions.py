import json
import os
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq()
MODEL = "openai/gpt-oss-120b"

def generate_questions():
    prompt = """
    Generate 100 math questions categorized into three sets: "simple calculation", "medium calculations", and "complex calculations".
    
    The questions must be ONLY calculation-based. 
    Topics must include:
    - Newton-Raphson Method: Finding roots of non-linear equations. Specify that the solution requires 6 to 7 iterations. Provide the function and an initial guess.
    - Numerical Trigonometry: Evaluate values like cos(64°), sin(22°), etc., to 4 decimal places.
    - Linear Algebra: Finding the adjoint of a 3x3 or 2x2 matrix (Mat_A). Provide the elements of the matrix.
    - Advanced Arithmetic: Multi-step numerical calculations involving powers, roots, and fractions.
    
    CRITICAL CONSTRAINTS:
    - NO proving questions (e.g., "Prove that...", "Show that...").
    - NO simplification questions (e.g., "Simplify the expression...").
    - ONLY numerical results or specific mathematical objects (like a matrix) should be the answer.
    
    Format the output as a JSON object with a single key 'questions', which is a list of objects:
    [
        {
            "id": 1,
            "difficulty": "simple",
            "topic": "numerical_trigonometry",
            "question": "Find the value of cos(64°) to four decimal places."
        },
        ...
    ]
    
    Ensure:
    - 33 Simple questions
    - 33 Medium questions
    - 34 Complex questions
    - Total 100 questions.
    - Questions should be clear and solvable.
    
    Output ONLY THE JSON. NO MARKDOWN FENCES.
    """
    
    print("Generating 100 math questions...")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=8000,
        stream=False,
    )
    
    content = response.choices[0].message.content.strip()
    
    # Clean up if any markdown fences are present
    if content.startswith("```json"):
        content = re.sub(r"^```json\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    elif content.startswith("```"):
        content = re.sub(r"^```\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        
    try:
        data = json.loads(content)
        with open("math_questions.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"Successfully generated {len(data['questions'])} questions and saved to math_questions.json")
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        print("Raw response saved to raw_response.txt")
        with open("raw_response.txt", "w", encoding="utf-8") as f:
            f.write(content)

if __name__ == "__main__":
    generate_questions()
