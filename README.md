# 🧭 Life Event Navigator

> An agentic AI planning team that turns a complex life goal into an evidence-aware, dependency-aware journey, and recalculates it when circumstances change.

Most chatbots answer one question at a time. **Life Event Navigator** behaves like a small team of specialists: it analyzes your goal, researches it, breaks it into phases, maps dependencies, stress-tests the plan, challenges its own assumptions, and then adapts when something changes (a delayed visa, a lower budget, a postponed admission).

Built for a hackathon with **Streamlit + CrewAI + Groq + FAISS**.

---

## ✨ Features

- **Multi-agent planning**: seven specialized CrewAI agents collaborate in a sequential pipeline, each building on the previous agent's output.
- **Document-aware (RAG)**: upload PDFs (admission letters, university guides, checklists). They are chunked, embedded and indexed in FAISS so agents can retrieve relevant passages.
- **Live web research**: agents search the public web (DuckDuckGo via `ddgs`) and are instructed to prefer official, first-party sources.
- **Dependency analysis**: explicit `A -> B because...` mapping, critical path, parallel tasks, and tasks that should *not* be done too early.
- **Risk & budget framework**: risk register with likelihood, impact, affected tasks and mitigation, without fabricated prices.
- **Devil's Advocate review**: a skeptical agent hunts for missing tasks, weak evidence and unrealistic assumptions.
- **Adaptive replanning**: describe a change ("my visa is delayed by 4 weeks") and the Replanner returns the impact on tasks, dependencies, timeline, risks and next actions.
- **Honest by design**: assumptions are labeled, high-impact requirements are flagged for verification, and the human stays in control.

---

## 🤖 Agent Team

| Agent | Responsibility |
|---|---|
| 🎯 **Life Goal Analyst** | Extracts event type, origin/destination, constraints, timeline and missing information |
| 🔎 **Research & Evidence Specialist** | Searches uploaded PDFs (FAISS) and the web; reports sources, URLs, confidence |
| 🗺️ **Journey Planning Specialist** | Builds 5-8 phases with tasks, timing, priority and completion criteria |
| 🔗 **Dependency Analyst** | Identifies blockers, prerequisites, critical path and parallel work |
| ⚠️ **Risk & Budget Analyst** | Builds the risk register and a budget framework |
| 🧨 **Devil's Advocate** | Challenges the plan and proposes alternatives |
| ♻️ **Adaptive Replanner** | Produces the final plan and re-plans when circumstances change |
| 🧑‍✈️ **Supervisor** | Coordinating role defined in the crew (see [Roadmap](#-roadmap)) |

### Pipeline

```
User goal + constraints (+ optional PDFs)
        │
        ▼
 Goal Analyst ─► Research ─► Planner ─► Dependency Analyst ─► Risk & Budget ─► Devil's Advocate ─► Final Plan
                    ▲
        FAISS (PDFs) + Web search
```

### Adaptive replanning

```
Existing plan + "Visa delayed by 6 weeks"
        │
        ▼
 Adaptive Replanner
        │
        ▼
 Change Impact · Affected Tasks · Updated Dependencies
 Updated Timeline · Updated Risks · New Next Actions · What to Verify
```

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Agent framework | CrewAI |
| LLM | Groq API, `llama-3.3-70b-versatile` (via LiteLLM) |
| Vector search | FAISS (`IndexFlatIP` with normalized embeddings) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| PDF parsing | pypdf |
| Web search | ddgs |

---

## 📁 Project Structure

```
life-event-navigator/
├── app.py                      # Streamlit app, agents, tasks, RAG, replanning
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── secrets.toml.example    # Template for your API key
```

---

## 🚀 Getting Started (Local)

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/life-event-navigator.git
cd life-event-navigator

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> Use a Python version supported by CrewAI (3.10 to 3.13).

### 2. Add your Groq API key

Get a free key at [console.groq.com](https://console.groq.com), then create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Alternatively, set it as an environment variable:

```bash
export GROQ_API_KEY="your_groq_api_key_here"      # Windows PowerShell: $env:GROQ_API_KEY="..."
```

### 3. Run

```bash
streamlit run app.py
```

The first run downloads the embedding model, so it may take a moment.

---

## ☁️ Deploy to Streamlit Community Cloud

1. Push the project to a **public GitHub repository**. Make sure `.streamlit/secrets.toml` is in `.gitignore` and **never committed**.
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
3. Select your repository, branch, and set the main file path to `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
5. Click **Deploy**.

---

## 🕹️ How to Use

1. **(Optional)** Upload supporting PDFs in the sidebar and click **📥 Index PDFs**.
2. Describe your life event and fill in location, target date, budget and travelers.
3. Click **🚀 Create My Life Journey**. The agents run in sequence (this can take a few minutes).
4. Review the generated plan, which includes:
   Life Event Summary · Journey Phases · Critical Dependencies · Next 5 Actions · Timeline · Budget Framework · Risks and Mitigations · Evidence and Sources · What You Should Verify · If Something Changes
5. Under **♻️ Something Changed?**, describe a change and click **🔄 Recalculate My Journey**. The impact analysis is appended to your plan.

### Example prompt

> I want to move from Pakistan to Canada for a Master's degree in September 2027. I have a budget of USD 20,000 and will travel alone.

### Example changes to try

- "My visa appointment was delayed by four weeks."
- "My budget decreased by 30%."
- "My admission was deferred to January 2028."

---

## ⚠️ Limitations

- **This is a prototype.** AI output can be incomplete or wrong. Always verify visa, legal, financial and academic requirements with official sources before acting.
- Web search returns snippets, not full pages; agents are told not to treat them as guaranteed truth.
- Uploaded PDFs are held in session memory and the FAISS index is rebuilt per query. Nothing is persisted between sessions.
- Scanned (image-only) PDFs have no extractable text and are not supported.
- Generation speed and quality depend on Groq's rate limits and the model's context window.

---

## 🗺️ Roadmap

- Visual dependency graph
- Timeline / Gantt view
- Live agent activity panel
- Inline source citations
- Persistent user memory
- Advanced budgeting
- **What-If Simulator** ("What if my budget drops by 30%?")
- Wire the Supervisor agent into a hierarchical crew process

---

## 🔒 Security

- Never hard-code or commit your API key. Use Streamlit Secrets or environment variables.
- Keep `.streamlit/secrets.toml` in `.gitignore`.

---

## 💡 Pitch

*Life Event Navigator is an Agentic AI operating system for major life transitions, turning complex goals into dependency-aware, evidence-supported, adaptive journeys.*

---

## 📄 License

Add a license of your choice (e.g. MIT) before publishing.
