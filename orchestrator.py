# orchestrator.py
from agents.websearch_agent import run_websearch_agent
from agents.power_flow_agent import run_power_flow_agent
from agents.matlab_executor_agent import run_matlab_executor_agent
from dotenv import load_dotenv
import os
from groq import Groq
import json
import base64

agent = Groq()

load_dotenv()

tools = [
  {
    "type": "function",
    "function": {
      "name": "route_query",
      "description": "Routes the query to the appropriate handler based on the type",
      "parameters": {
        "type": "object",
        "properties": {
          "type": {
            "type": "string",
            "enum": ["power_flow", "web_search", "matlab_executor"],
            "description": "The target handler for the query"
          },
          "query": {
            "type": "string",
            "description": "The actual query to be processed"
          }
        },
        "required": ["type", "query"]
      }
    }
  }
]

def classify_query(user_query, image_base64=None, conversation_history=None):
    """Classify user query using simple keyword-based method. Supports images and conversation history."""
    # Build user message content - add image if present
    user_content = f"{user_query}"
    if image_base64:
        user_content = [
            {
                "type": "text",
                "text": user_query
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                }
            }
        ]
    
    # Build messages list with system prompt, conversation history, and current query
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert agent that routes user queries to the appropriate handler. "
                "If an image is provided, analyze it and use it to help classify the query. "
                "Use the conversation history to understand context and follow-up questions. "
                "\n\n"
                "Route queries as follows:\n"
                "1. type = 'power_flow': For power flow analysis questions like solving Ybus, calculating bus voltages, "
                "admittance matrices, or power system network analysis. Ensure all data is in the query itself.\n"
                "2. type = 'matlab_executor': For general MATLAB programming tasks, control systems, signal processing, "
                "plotting, simulations, or any task requiring custom MATLAB code execution. This includes transfer functions, "
                "step responses, bode plots, state-space models, differential equations, etc.\n"
                "3. type = 'web_search': For general knowledge questions not related to technical computation.\n"
                "\n"
                "For small talk and greetings, DO NOT call any tool - just respond naturally."
            )
        }
    ]
    
    # Add conversation history if available
    if conversation_history:
        messages.extend(conversation_history)
    
    # Add current user query
    messages.append({
        "role": "user",
        "content": user_content
    })
    
    respone = agent.chat.completions.create(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        messages=messages,
        tools=tools,
        max_tokens=2000,
        stream=False
    )
    
    if respone.choices[0].message.tool_calls:
        tool_response = respone.choices[0].message.tool_calls[0].function
        print(tool_response)
        if tool_response.name == "route_query":
            arguments = json.loads(tool_response.arguments)
            print(arguments)
            return arguments['type'], arguments['query']
        else:
            return {"error": "Unexpected tool call"}
    else:
        response_content = respone.choices[0].message.content
        if response_content:
            return response_content.strip(), None
        return None, None

def orchestrate(user_query, image_base64=None, conversation_history=None):
    """Orchestrate query handling with optional image support and conversation history."""
    answer, query = classify_query(user_query, image_base64, conversation_history)
    print(f"Classified query as: {answer}")
    if answer == "power_flow":
        return run_power_flow_agent(query)
    elif answer == "web_search":
        return run_websearch_agent(query)
    elif answer == "matlab_executor":
        return run_matlab_executor_agent(query)
    else:
        return answer

def main():
    while True:
        user_query = input("User: ")
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        if not user_query:
            print("Please enter a valid query.")
            continue
        try:
            result = orchestrate(user_query)
            print(f"Response: {result}\n")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()