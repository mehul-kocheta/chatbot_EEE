# orchestrator.py
from gs_agent import run_power_flow_agent
from websearch_agent import run_websearch_agent
from loss_agent import run_loss_agent
from dotenv import load_dotenv
import os
from groq import Groq

agent = Groq()

load_dotenv()

tools = [
  {
    "type": "function",
    "function": {
      "name": "route_query",
      "description": "Routes the query to either the power_flow or web_search handler based on the type",
      "parameters": {
        "type": "object",
        "properties": {
          "type": {
            "type": "string",
            "enum": ["power_flow", "web_search", "loss_calculation_new_load"],
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

def classify_query(user_query):
    """Classify user query using simple keyword-based method."""
    respone = agent.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert agent that determines whether a user query "
                    "is related to POWERFLOW analysis or a GENERAL WEB SEARCH. "
                    "Call the route_query tool with type = 'power_flow' for questions about power flow analysis like solving the Ybus for volatage at each bus,"
                    "Call the route_query tool with type = 'loss_calculation_new_load' for questions about calculating power system losses after adding new loads,"
                    "bus voltages, admittance matrices, or similar topics. "
                    "Respond with 'web_search' for all other general knowledge queries."
                    "Else for small talk and greetings, never call any tool and just respond accordingly."
                )
            },
            {
                "role": "user",
                "content": f"{user_query}"
            }
        ],
        tools=tools,
        max_tokens=2000,
        stream=False
    )
    
    if respone.choices[0].message.tool_calls:
        tool_response = respone.choices[0].message.tool_calls[0]
        if tool_response.name == "route_query":
            arguments = tool_response.arguments
            return arguments['type'], arguments['query']
        else:
            return {"error": "Unexpected tool call"}
    else:
        return respone.choices[0].message.content.strip(), None

def orchestrate(user_query):
    answer, query = classify_query(user_query)
    if answer == "power_flow":
        return run_power_flow_agent(query)
    elif answer == "web_search":
        return run_websearch_agent(query)
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