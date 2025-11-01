# orchestrator.py
from gs_agent import run_power_flow_agent
from websearch_agent import run_websearch_agent

# Add or update these keywords as needed for power system queries
POWERFLOW_KEYWORDS = [
    "bus voltage", "ybus", "admittance matrix", "power flow", "gauss-seidel", "power system", "load flow", "transmission line",
    "node voltage", "power injections", "bus current", "slack bus", "PQ bus", "PV bus", "bus impedance"
]

def classify_query(user_query):
    """Classify user query using simple keyword-based method."""
    query_lower = user_query.lower()
    for kw in POWERFLOW_KEYWORDS:
        if kw in query_lower:
            return "POWERFLOW"
    return "WEBSEARCH"

def orchestrate(user_query):
    print("\nOrchestrator: Analyzing your query...")
    agent_type = classify_query(user_query)
    print(f"Routing to {agent_type} Agent.")
    print("="*60)
    if agent_type == "POWERFLOW":
        print("Power Flow Agent is processing your request...\n" + "="*60)
        return run_power_flow_agent(user_query)
    else:
        print("Web Search Agent is processing your request...\n" + "="*60)
        return run_websearch_agent(user_query)

def main():
    print("="*60)
    print("MULTI-AGENT ORCHESTRATOR")
    print("="*60)
    print("Capabilities:\n- Power Flow Analysis (Ybus, Bus Voltages, etc.)\n- Web Search (General questions)")
    print("="*60)
    while True:
        user_query = input("Enter your query (or 'quit' to exit): ").strip()
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        if not user_query:
            print("Please enter a valid query.")
            continue
        try:
            result = orchestrate(user_query)
            print("="*60)
            print("RESULT")
            print("="*60)
            print(result)
            print("="*60)
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()