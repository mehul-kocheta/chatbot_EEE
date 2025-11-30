# Architecture Components Guide
## Power System Analysis Chatbot - Component Overview

---

## 🏗️ System Components (One-Line Descriptions)

### **Frontend Layer**
- **Streamlit Web UI** (`app.py`): Chat-based web interface with image upload support and conversation history management

### **Orchestrator Layer**
- **Query Router** (`orchestrator.py`): Uses LLM with function calling to classify queries and route to appropriate agent (power_flow, web_search, or matlab_code)

### **Main Agents**

1. **Power Flow Agent** (`agents/power_flow_agent.py`): Master coordinator that orchestrates specialized sub-agents using tool calling for multi-step power system analysis workflows

2. **Web Search Agent** (`agents/websearch_agent.py`): Searches DuckDuckGo, scrapes web content, and uses LLM to synthesize comprehensive answers with sources

3. **MATLAB Executor Agent** (`agents/matlab_executor_agent.py`): Executes custom MATLAB code with dual modes - extracts data for matplotlib plotting or returns calculation results

### **Specialized Sub-Agents**

4. **Ybus Agent** (`agents/ybus_agent.py`): Parses line/branch data from natural language using LLM and calculates bus admittance matrix via MATLAB

5. **Gauss-Seidel Agent** (`agents/gs_agent.py`): Pure Python iterative power flow solver supporting PV/PQ buses with Q-limit checking and convergence tracking

6. **Loss Agent** (`agents/loss_agent.py`): Calculates total system power losses after load changes using MATLAB integration with Ybus and voltage profile

7. **Fault Agent** (`agents/fault_agent.py`): Analyzes three-phase bolted faults using MATLAB to compute fault currents and post-fault voltages/currents

### **External Services**

- **MATLAB Computation Layer**: Python-MATLAB bridge providing high-performance numerical computations for power system analysis

- **DuckDuckGo API**: Free web search service (no API key required) for knowledge retrieval and research queries

- **Groq API**: Fast LLM inference using Llama-3.3-70b for orchestration and GPT-OSS-120b for synthesis with function calling support

---

## 🔄 Quick Data Flow

```
User Query → Streamlit → Orchestrator (classifies) → Routes to Agent → 
Agent processes (calls sub-agents/MATLAB/web) → Returns result → Display
```

---

## 🎯 Routing Logic

| Query Type | Routes To | Example |
|------------|-----------|---------|
| Ybus/Power Flow/Loss/Fault | Power Flow Agent | "Calculate Ybus for 3 buses" |
| Transfer Function/Plotting/Differential Eq | MATLAB Executor | "Plot Bode diagram of H(s)=1/(s+1)" |
| General Knowledge/Research | Web Search Agent | "What is Newton-Raphson method?" |

---

**Last Updated**: November 30, 2025
