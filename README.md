# Autonomous Rsearch critic and synthesizer (Open-Source, Local LLMs, Powered by LangGraph)

This project is a multi-step **research orchestration engine** built using **LangGraph** and **open-source local models** (Ollama).  
It transforms a user query into structured subqueries, runs parallel research steps, evaluates the outputs for quality, checks for contradictions, and finally synthesizes a polished answer.

The system uses free, local models only — no OpenAI API or paid services required.

---

## 🚀 Features

### **Parallel Research**
- Each subquery is processed independently using:
  - `research_unit`
  - `critic_unit`
  - `factcheck_unit`
- LangGraph maps these nodes across all subqueries simultaneously.

### **Quality-Control Pipeline**
Each research output undergoes:
1. **Critic evaluation** (clarity and relevance)  
2. **Fact-check evaluation** (internal evidence weaknesses)  
3. **Contradiction check** (global consistency across outputs)

Weak outputs trigger automatic retries.

### **Modular Architecture**
Nodes are fully isolated and easy to extend:
- `research_unit`
- `critic_unit`
- `factcheck_unit`
- `contradiction`
- `synthesize`

Future nodes (planned Phase 3):
- Web search integration
- RAG-based fact validation
- Supervisor agent for subquery refinement
- Logging/visualization

---

## 🧠 How It Works (High-Level Flow)
User Query
↓
Expand into subqueries
↓
Parallel Research (map)
research_unit → critic_unit → factcheck_unit
↓ (fan-in)
Postprocess results
↓
Contradiction analysis
↓
Synthesis of final answer
↓
Output
---

## 🛠 Installation

### Install dependencies:

pip install -r requirements.txt


### Install Ollama (for local LLM inference):
https://ollama.com

Pull a model:

ollama pull llama3.1


---

## ▶️ Run the pipeline

python -m src.run --query "future of robotics in agriculture"


---

## 📂 Project Structure

src/
├── nodes/
│ ├── research_unit.py
│ ├── critic_unit.py
│ ├── factcheck_unit.py
│ ├── contradiction.py
│ ├── synthesize.py
├── state.py
├── graph_builder.py
└── run.py


---

## 🧩 About Models
All LLM calls use local free models via **Ollama**:


You can swap in any other Ollama model by editing the `model="..."` lines in the node files.

---

Export path using:-

# $env:PYTHONPATH = "$PWD;$env:PYTHONPATH"  ---> For Windows



## 📌 Next Steps (Planned Features)

Phase 3 will add:
- Web search tool integration
- Local RAG (vector search)
- Evidence-aware fact-checking
- Supervisor node for intelligent subquery refinement
- Graph visualization via LangGraph’s inspector

---

## 🧑‍💻 Author
Mayank Singh — Building intelligent research systems with LangGraph and open-source LLMs.
