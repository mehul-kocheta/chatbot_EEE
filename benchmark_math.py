import json
import os
import sys
import time
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv

# Add the chatbot directory to sys.path so we can import the agent
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from agents.matlab_executor_agent import run_matlab_executor_agent

load_dotenv()

# Clients
groq_client = Groq()
openai_client = OpenAI() # Assumes OPENAI_API_KEY is in environment

def get_gpt_reasoning_answer(question):
    """
    Get answer from GPT-5.4 with high reasoning effort.
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-5.4", # Flagship reasoning model
            messages=[
                {
                    "role": "user", 
                    "content": f"Solve this math problem step-by-step: {question}"
                }
            ],
            # GPT-5.4 specific parameter for tuning deliberation depth
            # Options: "none", "low", "medium", "high", "xhigh"
            reasoning_effort="high",
            max_completion_tokens=10000 
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI Error: {e}"


def run_benchmark():
    if not os.path.exists("math_questions.json"):
        print("math_questions.json not found. Run generate_math_questions.py first.")
        return

    with open("math_questions.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data["questions"]
    results = []

    print(f"Starting benchmark for {len(questions)} questions...")

    for i, q in enumerate(questions):
        print(f"[{i+1}/100] ID {q['id']} ({q['difficulty']} - {q['topic']})")
        
        question_text = q["question"]
        
        # Get Agent Answer (MATLAB) with retries
        print("  Running Agent (MATLAB)...")
        start_time_agent = time.time()
        
        MAX_RETRIES = 3
        agent_answer = ""
        for attempt in range(MAX_RETRIES):
            try:
                agent_answer = run_matlab_executor_agent(question_text)
                # Check if the response indicates an agent error
                if "Agent Error:" not in str(agent_answer):
                    break
                print(f"    Attempt {attempt + 1} failed: {agent_answer}. Retrying...")
            except Exception as e:
                agent_answer = f"Agent Error: {e}"
                print(f"    Attempt {attempt + 1} failed with exception: {e}. Retrying...")
            
            if attempt < MAX_RETRIES - 1:
                time.sleep(2) # Wait 2 seconds before retrying
        
        end_time_agent = time.time()
        agent_time = end_time_agent - start_time_agent
        
        # Get GPT Answer (OpenAI GPT-5.4)
        print("  Running GPT-5.4...")
        start_time_gpt = time.time()
        gpt_answer = get_gpt_reasoning_answer(question_text)
        end_time_gpt = time.time()
        gpt_time = end_time_gpt - start_time_gpt
        
        result = {
            "id": q["id"],
            "difficulty": q["difficulty"],
            "topic": q["topic"],
            "question": question_text,
            "agent_answer": agent_answer,
            "agent_time_seconds": round(agent_time, 2),
            "gpt_answer": gpt_answer,
            "gpt_time_seconds": round(gpt_time, 2)
        }
        results.append(result)
        
        # Periodic save
        if (i + 1) % 5 == 0:
            with open("benchmark_results_partial.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)
            print(f"  Progress saved to benchmark_results_partial.json")

    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    
    print("Benchmark complete! Results saved to benchmark_results.json")

if __name__ == "__main__":
    run_benchmark()
