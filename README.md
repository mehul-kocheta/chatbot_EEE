# Power System Analysis Chatbot

An intelligent multi-agent chatbot system for power system analysis tasks, leveraging Large Language Models (LLMs) and MATLAB integration for complex electrical engineering computations.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2024a+-orange.svg)](https://www.mathworks.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.51.0-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Table of Contents

- [Features](#-features)
- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Technology Stack](#️-technology-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

- **Ybus Matrix Calculation** - Computing bus admittance matrices from branch data
- **Power Flow Analysis** - Solving load flow using Gauss-Seidel method
- **System Loss Calculation** - Computing total system losses after load changes
- **Fault Analysis** - Analyzing three-phase bolted faults
- **MATLAB Code Execution** - General-purpose MATLAB code generation and execution for control systems, signal processing, and mathematical computations
- **General Web Search** - Answering broader power system questions via web search
- **Multimodal Input Support** - Text and image inputs for enhanced analysis
- **Intelligent Query Routing** - Automatic classification and routing to appropriate agents

## 🎯 Project Overview

This project implements an AI-powered assistant capable of handling various power system analysis tasks. The system uses a sophisticated multi-agent architecture with intelligent query routing and supports multimodal inputs (text + images).

The chatbot leverages:
- **Groq API** for fast LLM inference
- **MATLAB Engine API** for numerical computations
- **Streamlit** for web interface
- **Multi-agent pattern** for specialized task handling

## 🏗️ Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend Layer                            │
│                    (Streamlit Web UI)                            │
│              - Chat Interface                                    │
│              - Image Upload Support                              │
│              - Conversation History                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Orchestrator Layer                            │
│                   (orchestrator.py)                              │
│                                                                   │
│  ┌──────────────────────────────────────────────────┐           │
│  │ LLM Query Classifier (Groq + Llama)              │           │
│  │ - Analyzes user intent                           │           │
│  │ - Routes to appropriate agent                    │           │
│  │ - Handles conversation context                   │           │
│  │ - Supports multimodal input (text + images)      │           │
│  └──────────────────────────────────────────────────┘           │
│                         │                                         │
│            ┌────────────┴────────────┬──────────────┐           │
│            ▼                           ▼              ▼           │
│   ┌─────────────────┐    ┌──────────────────┐  ┌──────────────┐  │
│   │ Power Flow      │    │ MATLAB Executor │  │ Web Search   │  │
│   │ Agent           │    │ Agent           │  │ Agent        │  │
│   └─────────────────┘    └────────┬─────────┘  └──────────────┘  │
└────────────────────────────────────┼───────────────────────────────┘
                         │            │                    │
                         ▼            │                    ▼
┌─────────────────────────────────────────┐  ┌──────────────────┐
│     Power System Sub-Agents             │  │  DuckDuckGo API  │
│                                         │  │  + LLM Synthesis │
│  ┌──────────────────────────────────┐  │  └──────────────────┘
│  │ Ybus Agent (ybus_agent.py)      │  │
│  │ - Parses branch data             │  │
│  │ - Calls MATLAB for computation   │  │
│  └──────────────────────────────────┘  │
│                                         │
│  ┌──────────────────────────────────┐  │
│  │ Gauss-Seidel Agent (gs_agent.py)│  │
│  │ - Power flow solver              │  │
│  │ - Handles PV and PQ buses        │  │
│  │ - Pure Python implementation     │  │
│  └──────────────────────────────────┘  │
│                                         │
│  ┌──────────────────────────────────┐  │
│  │ Loss Agent (loss_agent.py)       │  │
│  │ - System loss calculation        │  │
│  │ - Uses MATLAB integration        │  │
│  └──────────────────────────────────┘  │
│                                         │
│  ┌──────────────────────────────────┐  │
│  │ Fault Agent (fault_agent.py)     │  │
│  │ - Bolted fault analysis          │  │
│  │ - Post-fault voltage/current     │  │
│  │ - Uses MATLAB integration        │  │
│  └──────────────────────────────────┘  │
└─────────────────────────────────────────┘
                    │            │
                    ▼            │
┌─────────────────────────────────────────┐
│      MATLAB Computation Layer           │
│                                         │
│  ┌──────────────────────────────────┐  │
│  │ MATLAB Engine API                 │  │
│  │ - Dynamic code execution          │  │
│  │ - Workspace variable access       │  │
│  │ - Plot data extraction            │  │
│  └──────────────────────────────────┘  │
│                                         │
│  ┌──────────────────────────────────┐  │
│  │ MATLAB Scripts                   │  │
│  │ - calculate_fault.m              │  │
│  │ - calculate_loss.m               │  │
│  │ - gauss_siedel_easy.m            │  │
│  │ - And more...                    │  │
│  └──────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

### Key Components

#### 1. **Orchestrator** (`orchestrator.py`)
The central routing system that:
- Uses Groq's LLM with tool calling to classify user queries
- Routes queries to one of three agents: power-flow, matlab_executor, or web-search
- Supports multimodal inputs (text + base64 encoded images)
- Maintains conversation history for context-aware responses
- Handles small talk and greetings without agent calls

#### 2. **Power Flow Agent** (`agents/power_flow_agent.py`)
Master coordinator for power system analysis:
- Orchestrates multiple specialized sub-agents
- Uses tool calling to chain operations (e.g., Ybus → Power Flow → Loss Calculation)
- Implements iterative conversation loop for complex multi-step problems
- Returns formatted markdown responses with disclaimers

#### 3. **Specialized Sub-Agents**

**Ybus Agent** (`agents/ybus_agent.py`)
- Parses branch/line data from natural language
- Extracts: from_bus, to_bus, R, X, transformer ratio (a), shunt admittance
- Uses MATLAB engine to compute bus admittance matrix
- Handles missing parameters with defaults

**Gauss-Seidel Agent** (`agents/gs_agent.py`)
- Pure Python power flow solver
- Supports PV buses (with Q-limits) and PQ buses
- Iterative solver with convergence tracking
- Computes power injections and system losses
- Returns voltages in rectangular and polar forms

**Loss Agent** (`agents/loss_agent.py`)
- Calculates total system power loss
- Takes Ybus, voltage profile, and new load details
- Uses MATLAB integration for computation
- Formula: `Ploss = real(sum(V .* conj(Ybus * V)))`

**Fault Agent** (`agents/fault_agent.py`)
- Analyzes three-phase bolted faults
- Accepts Ybus or Zbus matrices
- Computes post-fault voltages and currents
- Uses MATLAB scripts for fault calculations

#### 4. **MATLAB Executor Agent** (`agents/matlab_executor_agent.py`)
Intelligent MATLAB code generation and execution system:
- **LLM-Powered Code Generation** - Uses GPT-OSS-120B to generate MATLAB code from natural language queries
- **Dual Execution Modes**:
  - **Calculation Mode**: Executes MATLAB code and returns text output (e.g., matrix operations, numerical computations)
  - **Plotting Mode**: Extracts plot data from MATLAB workspace and uses matplotlib to generate visualizations
- **Smart Code Analysis** - Automatically determines if a task requires plotting or calculation
- **Plot Data Extraction** - Supports multiple plot formats (x_data/y_data, x1/y1/x2/y2, or common variable names)
- **Metadata Support** - Handles plot titles, labels, and legends from MATLAB workspace
- **Error Handling** - Provides detailed error messages for debugging
- **Iterative Refinement** - Uses tool calling with up to 5 iterations for complex tasks

**Key Features:**
- Generates MATLAB code for control systems, signal processing, and mathematical computations
- For plotting tasks: Extracts data from MATLAB and creates matplotlib visualizations
- For calculation tasks: Executes MATLAB code and returns formatted text output
- Returns formatted responses with code, output, and plots

#### 5. **Web Search Agent** (`agents/websearch_agent.py`)
- Uses DuckDuckGo API for web searches
- Fetches top 20 results
- Synthesizes answers using Groq LLM
- Provides sources and URLs

#### 6. **Frontend** (`app.py`)
Streamlit-based web interface with:
- Chat-style conversation UI
- Image upload support with preview
- Session-based conversation history
- Responsive design with custom CSS
- Loading spinners and status indicators

#### 7. **MATLAB Integration Layer**
Python-MATLAB bridge for numerical computations:
- `fault_analysis_matlab.py` - Fault analysis wrapper
- `loss_after_new_load.py` - Loss calculation wrapper
- `matlab_scripts/` - Collection of MATLAB functions
- `agents/matlab_executor_agent.py` - General-purpose MATLAB code executor

---

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.10+** - Primary programming language
- **Groq API** - LLM inference (Llama and GPT-OSS models)
- **MATLAB Engine API** - Numerical computations
- **NumPy** - Array operations and complex number handling

### Agent Framework
- **Tool Calling** - LLM-based function calling for agent coordination
- **Multi-Agent Pattern** - Hierarchical agent architecture

### Web & Search
- **Streamlit** - Web interface framework
- **DuckDuckGo (ddgs)** - Web search API
- **Pillow** - Image processing

### Development Tools
- **python-dotenv** - Environment variable management
- **Git** - Version control

---

## 📦 Installation

### Prerequisites
1. **Python 3.10 or higher**
2. **MATLAB R2024a or higher** (with valid license)
3. **MATLAB Engine API for Python** installed
4. **Groq API key** (get from [https://console.groq.com](https://console.groq.com))

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Major_Project.git
cd Major_Project
```

2. **Create and activate virtual environment**
```bash
python -m venv myenv

# On Windows
myenv\Scripts\activate

# On Linux/Mac
source myenv/bin/activate
```

3. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

4. **Install MATLAB Engine API**
```bash
# Navigate to MATLAB engine setup directory
cd "matlabroot\extern\engines\python"

# Install the engine
python setup.py install
```

5. **Configure environment variables**
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```

6. **Verify MATLAB scripts**
Ensure all MATLAB scripts are in `matlab_scripts/` directory

---

## 🚀 Usage

### Running the Web Interface
```bash
streamlit run app.py
```
The application will open in your browser at `http://localhost:8501`

### Running the CLI Version
```bash
python orchestrator.py
```

### Example Queries

**Power Flow Analysis**:
```
Calculate bus voltages for a 3-bus system with the following data:
- Branch 1: From bus 1 to bus 2, R=0.03, X=0.08, shunt=0.04
- Branch 2: From bus 1 to bus 3, R=0.02, X=0.05, shunt=0.02
- Bus 1: Slack bus, V=1.0∠0°
- Bus 2: PQ bus, P=-1.5, Q=-0.5
- Bus 3: PV bus, P=-2.0, V=1.02
```

**Fault Analysis**:
```
Find post-fault voltages for a three-phase fault at bus 2.
Pre-fault voltages: [1.0+0j, 0.95-0.1j, 0.98-0.05j]
Ybus: [[10-20j, -5+10j, -5+10j], ...]
```

**Loss Calculation**:
```
Calculate system losses after adding a load of 0.5+0.2j pu at bus 3.
Current voltages: [1.0, 0.95∠-5°, 0.98∠-3°]
```

**MATLAB Code Execution**:
```
Plot the step response of the transfer function H(s) = 5 / (s^2 + 3s + 2)
```

```
Create a 3x3 matrix with values [[1,2,3],[4,5,6],[7,8,9]] and calculate its determinant and eigenvalues.
```

**Web Search**:
```
What is the difference between Newton-Raphson and Gauss-Seidel methods?
```

### Using Image Input
1. Click the 📎 "Attach image" button
2. Upload a circuit diagram or system schematic
3. Ask a question about the image
4. The system will analyze the image and provide context-aware responses

---

## 📂 Project Structure

```
Major_Project/
├── agents/                          # Agent modules
│   ├── __init__.py
│   ├── power_flow_agent.py         # Master power flow coordinator
│   ├── ybus_agent.py                # Ybus matrix calculator
│   ├── gs_agent.py                  # Gauss-Seidel solver
│   ├── loss_agent.py                # Loss calculator
│   ├── fault_agent.py               # Fault analyzer
│   ├── matlab_executor_agent.py     # MATLAB code generation & execution
│   └── websearch_agent.py           # Web search agent
│
├── matlab_scripts/                  # MATLAB computation scripts
│   ├── calculate_fault.m
│   ├── calculate_loss.m
│   ├── gauss_siedel_easy.m
│   ├── gauss_siedel_easy_2.m
│   ├── NR_easy.m
│   ├── NR_2.m
│   ├── lab_end_practice.m
│   ├── point_by_point.m
│   └── swing.m
│
├── orchestrator.py                  # Main orchestrator with query routing
├── app.py                           # Streamlit web interface
├── fault_analysis_matlab.py         # Python-MATLAB wrapper for faults
├── loss_after_new_load.py           # Python-MATLAB wrapper for losses
├── gs_solver.py                     # Standalone GS solver
│
├── requirements.txt                 # Python dependencies
├── .env                             # Environment variables (not in git)
├── .gitignore                       # Git ignore file
├── README.md                        # This file
├── YBUS_AGENT_README.md            # Ybus agent documentation
│
├── test.py                          # Test scripts
├── test1.py
├── test_ybus.py
│
├── architecture_major_project.png   # Architecture diagram
├── accuracy_vs_buses.png            # Analysis plots
├── combined_accuracy_analysis.png
├── mass-system.png
├── step-response.png
├── random.png
│
├── project_report.pdf               # Project documentation
├── project_report.tex              # LaTeX source for report
├── linkedin_post.md                # Social media post
└── EE403_Power System & Renewable Energy Lab_Student Manual.pdf
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 🙏 Acknowledgments

- Groq for providing fast LLM inference API
- MATLAB for numerical computation capabilities
- Streamlit for the web framework
- All contributors and users of this project

---

Made with ❤️ and ⚡ by the Power Systems Team
