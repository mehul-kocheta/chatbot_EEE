# Power System Analysis Chatbot

An intelligent multi-agent chatbot system for power system analysis tasks, leveraging Large Language Models (LLMs) and MATLAB integration for complex electrical engineering computations.

## 🎯 Project Overview

This project implements an AI-powered assistant capable of handling various power system analysis tasks including:
- **Ybus Matrix Calculation** - Computing bus admittance matrices from branch data
- **Power Flow Analysis** - Solving load flow using Gauss-Seidel method
- **System Loss Calculation** - Computing total system losses after load changes
- **Fault Analysis** - Analyzing three-phase bolted faults
- **General Web Search** - Answering broader power system questions via web search

The system uses a sophisticated multi-agent architecture with intelligent query routing and supports multimodal inputs (text + images).

---

## 🏗️ Project Architecture

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
│            ┌────────────┴──────────────┐                         │
│            ▼                            ▼                         │
│   ┌─────────────────┐        ┌──────────────────┐               │
│   │ Power Flow      │        │ Web Search       │               │
│   │ Agent           │        │ Agent            │               │
│   └─────────────────┘        └──────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
                         │                            │
                         ▼                            ▼
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
                    │
                    ▼
┌─────────────────────────────────────────┐
│      MATLAB Computation Layer           │
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
- Determines if query is power-flow related or web-search related
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

#### 4. **Web Search Agent** (`agents/websearch_agent.py`)
- Uses DuckDuckGo API for web searches
- Fetches top 20 results
- Synthesizes answers using Groq LLM
- Provides sources and URLs

#### 5. **Frontend** (`app.py`)
Streamlit-based web interface with:
- Chat-style conversation UI
- Image upload support with preview
- Session-based conversation history
- Responsive design with custom CSS
- Loading spinners and status indicators

#### 6. **MATLAB Integration Layer**
Python-MATLAB bridge for numerical computations:
- `fault_analysis_matlab.py` - Fault analysis wrapper
- `loss_after_new_load.py` - Loss calculation wrapper
- `matlab_scripts/` - Collection of MATLAB functions

---

## 🗓️ 6-Week Development Timeline

### **Week 1: Research & Planning Phase** 📚
**Objective**: Understand the problem domain and design the solution

**Activities**:
- ✅ Studied power system analysis fundamentals
  - Ybus matrix formulation
  - Power flow analysis methods (Gauss-Seidel, Newton-Raphson)
  - Fault analysis techniques
  - Loss calculation methods

- ✅ Researched available tools and technologies
  - Explored LLM options (OpenAI, Groq, Anthropic)
  - Evaluated MATLAB vs pure Python approaches
  - Investigated web search APIs
  - Reviewed chatbot frameworks

- ✅ Reviewed lab manual and requirements
  - Analyzed `EE403_Power System & Renewable Energy Lab_Student Manual.pdf`
  - Identified key computations needed
  - Listed typical student queries

- ✅ Designed multi-agent architecture
  - Conceptualized orchestrator pattern
  - Planned agent hierarchy and responsibilities
  - Defined communication protocols

- ✅ Created initial MATLAB scripts for validation
  - `gauss_siedel_easy.m`
  - `NR_easy.m`
  - Basic Ybus calculation scripts

**Deliverables**:
- Architecture design document
- Technology stack selection
- MATLAB validation scripts
- Project timeline and milestones

---

### **Week 2: Architecture & Core Setup** 🏗️
**Objective**: Set up the project foundation and implement the orchestrator

**Activities**:
- ✅ Set up development environment
  - Created Python virtual environment
  - Installed core dependencies (Groq, MATLAB Engine, numpy)
  - Set up `.env` for API key management

- ✅ Implemented the orchestrator system
  - Created `orchestrator.py` with query classification logic
  - Integrated Groq API with Llama model
  - Implemented tool calling for query routing
  - Added support for conversation history

- ✅ Built basic agent structure
  - Created `agents/` package with `__init__.py`
  - Defined agent interface patterns
  - Implemented basic error handling

- ✅ Added multimodal support
  - Integrated base64 image encoding/decoding
  - Modified orchestrator to handle image inputs
  - Updated system prompts for image analysis

- ✅ Version control setup
  - Initialized git repository
  - Created `.gitignore` for Python and MATLAB files

**Deliverables**:
- Working orchestrator with query classification
- Basic project structure
- Environment configuration files (`requirements.txt`, `.env`)

**Key Files Created**:
- `orchestrator.py`
- `agents/__init__.py`
- `requirements.txt`
- `.env.example`

---

### **Week 3: MATLAB Integration & Basic Agents** 🔧
**Objective**: Integrate MATLAB and implement core computational agents

**Activities**:
- ✅ Set up MATLAB-Python bridge
  - Installed MATLAB Engine API for Python
  - Created test scripts to verify connection
  - Handled data type conversions (NumPy ↔ MATLAB)

- ✅ Implemented Ybus Agent
  - Created `agents/ybus_agent.py`
  - Developed LLM-based branch data parser
  - Integrated MATLAB Ybus computation
  - Added validation and error handling
  - Tested with various line configurations

- ✅ Developed MATLAB computation scripts
  - Enhanced Ybus calculation algorithm
  - Created `calculate_fault.m` for fault analysis
  - Created `calculate_loss.m` for loss computation

- ✅ Created Python-MATLAB wrapper functions
  - `fault_analysis_matlab.py` - Fault analysis wrapper
  - `loss_after_new_load.py` - Loss calculation wrapper
  - Handled complex number conversions
  - Managed MATLAB engine lifecycle (start/stop)

- ✅ Testing and validation
  - Created `test_ybus.py` for unit tests
  - Validated against manual calculations
  - Documented test cases

**Deliverables**:
- Functional Ybus Agent with MATLAB integration
- MATLAB wrapper functions
- Test suite for Ybus calculations
- `YBUS_AGENT_README.md` documentation

**Key Files Created**:
- `agents/ybus_agent.py`
- `fault_analysis_matlab.py`
- `loss_after_new_load.py`
- `matlab_scripts/calculate_fault.m`
- `matlab_scripts/calculate_loss.m`
- `test_ybus.py`

---

### **Week 4: Advanced Agents & Orchestration** 🤖
**Objective**: Implement remaining specialized agents and power flow coordinator

**Activities**:
- ✅ Implemented Gauss-Seidel Agent
  - Created `agents/gs_agent.py`
  - Developed pure Python Gauss-Seidel solver
  - Added support for PV and PQ buses
  - Implemented Q-limit checking for PV buses
  - Added convergence tracking and iteration limits
  - Computed power injections and system losses

- ✅ Implemented Loss Agent
  - Created `agents/loss_agent.py`
  - Integrated with MATLAB loss calculation
  - Parsed complex load data from natural language
  - Added post-calculation analysis

- ✅ Implemented Fault Agent
  - Created `agents/fault_agent.py`
  - Handled both Ybus and Zbus inputs
  - Computed post-fault voltages and currents
  - Integrated with MATLAB fault analysis

- ✅ Built Power Flow Agent coordinator
  - Created `agents/power_flow_agent.py`
  - Implemented tool calling for sub-agent orchestration
  - Added support for multi-step workflows
  - Enabled chaining: Ybus → Power Flow → Loss/Fault
  - Implemented iterative conversation loop

- ✅ Enhanced orchestrator
  - Connected all agents to orchestrator
  - Refined query classification prompts
  - Added better context handling

**Deliverables**:
- Four fully functional specialized agents
- Power flow coordinator with tool chaining
- Enhanced orchestrator with all agents integrated

**Key Files Created**:
- `agents/gs_agent.py`
- `agents/loss_agent.py`
- `agents/fault_agent.py`
- `agents/power_flow_agent.py`

---

### **Week 5: Web Search & Frontend Development** 🌐
**Objective**: Add web search capability and build user interface

**Activities**:
- ✅ Implemented Web Search Agent
  - Created `agents/websearch_agent.py`
  - Integrated DuckDuckGo Search API (`ddgs`)
  - Configured to fetch top 20 results
  - Implemented LLM-based answer synthesis
  - Added fallback for LLM failures

- ✅ Developed Streamlit frontend
  - Created `app.py` with Streamlit
  - Built chat-style conversation interface
  - Implemented message history management
  - Added session state handling

- ✅ Added image upload support
  - Integrated file uploader widget
  - Added image preview functionality
  - Implemented base64 encoding for API calls
  - Created compact uploader UI with custom CSS

- ✅ Enhanced UI/UX
  - Added custom CSS styling
  - Created sidebar with project information
  - Added loading spinners and status indicators
  - Implemented responsive design
  - Added emoji icons for better visual appeal

- ✅ Conversation management
  - Implemented conversation history tracking
  - Added context passing to orchestrator
  - Handled multimodal message display (text + images)

**Deliverables**:
- Fully functional web search agent
- Complete Streamlit web application
- Multimodal chat interface
- User documentation in sidebar

**Key Files Created**:
- `agents/websearch_agent.py`
- `app.py`

---

### **Week 6: Testing, Integration & Deployment** ✅
**Objective**: Test the complete system, fix bugs, and prepare for deployment

**Activities**:
- ✅ End-to-end testing
  - Tested all power system analysis workflows
  - Validated Ybus → Power Flow → Loss chain
  - Tested fault analysis with various configurations
  - Verified web search functionality
  - Tested image upload with circuit diagrams

- ✅ Integration testing
  - Tested conversation context maintenance
  - Verified multimodal input handling
  - Checked agent handoffs and routing
  - Validated error handling across all agents

- ✅ Bug fixes and optimization
  - Fixed convergence issues in Gauss-Seidel
  - Improved LLM prompt engineering
  - Optimized MATLAB engine lifecycle
  - Enhanced error messages
  - Added input validation

- ✅ Documentation
  - Created comprehensive README
  - Added inline code documentation
  - Created test files (`test.py`, `test1.py`)
  - Documented API usage and examples

- ✅ Performance optimization
  - Reduced unnecessary MATLAB engine restarts
  - Optimized LLM token usage
  - Improved response formatting
  - Added caching for repeated queries

- ✅ Deployment preparation
  - Updated `requirements.txt` with all dependencies
  - Created environment setup instructions
  - Added disclaimers for LLM-generated results
  - Prepared demo scenarios

**Deliverables**:
- Fully tested and validated system
- Comprehensive documentation
- Deployment-ready application
- Demo materials and test cases

**Key Files Created**:
- `test.py`
- `test1.py`
- `README.md` (this file)
- Various plot outputs (`plot_temp.png`, etc.)

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
git clone <repository-url>
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
│   └── websearch_agent.py           # Web search agent
│
├── matlab_scripts/                  # MATLAB computation scripts
│   ├── calculate_fault.m
│   ├── calculate_loss.m
│   ├── gauss_siedel_easy.m
│   ├── NR_easy.m
│   └── ...
│
├── myenv/                           # Virtual environment (not in git)
│
├── orchestrator.py                  # Main orchestrator with query routing
├── app.py                           # Streamlit web interface
├── fault_analysis_matlab.py         # Python-MATLAB wrapper for faults
├── loss_after_new_load.py           # Python-MATLAB wrapper for losses
├── gs_solver.py                     # Standalone GS solver
│
├── requirements.txt                 # Python dependencies
├── .env                             # Environment variables (not in git)
├── README.md                        # This file
├── YBUS_AGENT_README.md            # Ybus agent documentation
│
├── test.py                          # Test scripts
├── test1.py
├── test_ybus.py
│
└── EE403_Power System & Renewable Energy Lab_Student Manual.pdf
```

---

## 🔑 Key Features

### 1. **Intelligent Query Routing**
- Automatically classifies queries as power-flow or web-search
- Context-aware routing based on conversation history
- Handles small talk and greetings naturally

### 2. **Multi-Step Workflows**
- Chains multiple agents for complex tasks
- Example: Branch Data → Ybus → Power Flow → Loss Calculation
- Iterative problem solving with tool calling

### 3. **Multimodal Support**
- Accepts text and image inputs simultaneously
- Analyzes circuit diagrams and schematics
- Context-aware image understanding

### 4. **MATLAB Integration**
- Seamless Python-MATLAB communication
- Automatic data type conversion
- Efficient engine lifecycle management

### 5. **Conversation Memory**
- Maintains chat history across queries
- Context-aware follow-up questions
- Session-based state management

### 6. **User-Friendly Interface**
- Clean chat-style UI
- Real-time response streaming
- Image preview and attachment
- Informative error messages

### 7. **Reliability & Safety**
- LLM result disclaimers on all outputs
- Comprehensive error handling
- Input validation and sanitization
- Convergence checking for iterative solvers

---

## ⚠️ Important Notes

### LLM-Generated Results
All results produced by this system are generated by Large Language Models and may contain errors. Users should:
- Verify results with manual calculations
- Cross-check critical computations
- Use results as guidance, not absolute truth
- Consult with domain experts for production use

### MATLAB Licensing
This project requires a valid MATLAB license. Ensure you have:
- MATLAB R2024a or higher installed
- Required toolboxes (Control System, Optimization)
- MATLAB Engine API for Python installed

### API Rate Limits
- Groq API has rate limits based on your plan
- Consider caching for frequently asked questions
- Monitor API usage to avoid unexpected costs

---

## 🐛 Known Issues & Limitations

1. **Convergence Issues**
   - Gauss-Seidel may not converge for poorly conditioned systems
   - Try better initial voltage guesses
   - Consider using Newton-Raphson for difficult cases

2. **MATLAB Engine Overhead**
   - Starting MATLAB engine takes 2-5 seconds
   - Repeated calls can be slow
   - Consider keeping engine alive for multiple queries

3. **Image Analysis**
   - LLM image understanding is limited
   - May not accurately read handwritten circuit diagrams
   - Best results with clean, digital schematics

4. **Web Search Accuracy**
   - Search results depend on DuckDuckGo availability
   - LLM synthesis may introduce biases
   - Always verify technical information

---

## 🔮 Future Enhancements

### Short Term
- [ ] Add Newton-Raphson power flow solver
- [ ] Implement economic dispatch calculations
- [ ] Add support for PDF document upload and parsing
- [ ] Create visualization for power flow results
- [ ] Add unit conversion utilities

### Medium Term
- [ ] Implement optimal power flow (OPF)
- [ ] Add stability analysis features
- [ ] Create database for storing calculations
- [ ] Add export functionality (PDF reports)
- [ ] Implement user authentication

### Long Term
- [ ] Real-time power system monitoring
- [ ] Integration with power system simulators (PSS/E, PowerWorld)
- [ ] Mobile application
- [ ] Multi-language support
- [ ] Cloud deployment with scaling

---

## 🤝 Contributing

This is a major project for academic purposes. For collaboration:
1. Fork the repository
2. Create a feature branch
3. Make your changes with clear documentation
4. Submit a pull request with detailed description

---

## 📄 License

This project is created for academic purposes as part of the EE403 Power System & Renewable Energy Lab course.

---

## 👥 Authors

**Power Systems Team**
- Project Lead & Developer: Mehul
- Course: EE403 - Power System & Renewable Energy Lab
- Institution: [Your Institution Name]

---

## 📞 Contact & Support

For questions, issues, or suggestions:
- Create an issue in the repository
- Email: [your-email@example.com]
- Office Hours: [Schedule]

---

## 🙏 Acknowledgments

- **Course Instructor**: For guidance and lab manual
- **Groq**: For providing LLM API access
- **MathWorks**: For MATLAB software
- **Streamlit**: For excellent web framework
- **Open Source Community**: For various libraries and tools

---

## 📚 References

1. Power System Analysis - Hadi Saadat
2. Modern Power System Analysis - I.J. Nagrath & D.P. Kothari
3. Groq API Documentation - https://console.groq.com/docs
4. MATLAB Documentation - https://www.mathworks.com/help/matlab/
5. Streamlit Documentation - https://docs.streamlit.io/

---

**Last Updated**: November 28, 2025

**Version**: 1.0.0

**Status**: ✅ Production Ready

---

Made with ❤️ and ⚡ by the Power Systems Team






