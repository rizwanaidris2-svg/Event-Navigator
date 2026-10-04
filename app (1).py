import os
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Type

import numpy as np
import streamlit as st
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from ddgs import DDGS

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


# ============================================================
# LIFE EVENT NAVIGATOR
# Streamlit + CrewAI + FAISS + Groq
# ============================================================

APP_TITLE = "Life Event Navigator"
MODEL_NAME = "llama-3.3-70b-versatile"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = Path("faiss_store")
INDEX_DIR.mkdir(exist_ok=True)


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    .agent-card {
        padding: 1rem;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        margin-bottom: .7rem;
        background: #fafafa;
    }
    .small {
        color: #6b7280;
        font-size: .85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">🧭 Life Event Navigator</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">A multi-agent AI system that turns a complex life goal into an adaptive, evidence-aware journey.</div>',
    unsafe_allow_html=True,
)


# ============================================================
# Secrets / environment
# ============================================================

def get_secret(name: str, default: str = "") -> str:
    """Read Streamlit secrets first, then environment variables."""
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)


GROQ_API_KEY = get_secret("GROQ_API_KEY")
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# ============================================================
# Session state
# ============================================================

DEFAULT_STATE = {
    "knowledge_chunks": [],
    "knowledge_sources": [],
    "life_event": None,
    "plan": None,
    "research": None,
    "last_run": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# Embeddings + FAISS
# ============================================================

@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_faiss_index(chunks: List[str]):
    if not chunks:
        return None

    embedder = get_embedder()
    embeddings = embedder.encode(
        chunks,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def retrieve_faiss(query: str, k: int = 5) -> str:
    chunks = st.session_state.get("knowledge_chunks", [])
    if not chunks:
        return "No user-provided documents are currently indexed."

    index = build_faiss_index(chunks)
    if index is None:
        return "No indexed knowledge is available."

    embedder = get_embedder()
    q = embedder.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    scores, ids = index.search(q, min(k, len(chunks)))

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0:
            source = (
                st.session_state["knowledge_sources"][idx]
                if idx < len(st.session_state["knowledge_sources"])
                else "Uploaded document"
            )
            results.append(
                f"[Source: {source} | similarity={float(score):.3f}]\n"
                f"{chunks[idx]}"
            )

    return "\n\n".join(results) if results else "No relevant document passages found."


# ============================================================
# Custom CrewAI tools
# ============================================================

class RAGSearchInput(BaseModel):
    query: str = Field(..., description="Question or information need to search in indexed user documents.")


class RAGSearchTool(BaseTool):
    name: str = "FAISS Document Search"
    description: str = (
        "Searches the user's uploaded documents using a FAISS vector index. "
        "Use this when the answer may be contained in uploaded PDFs or documents."
    )
    args_schema: Type[BaseModel] = RAGSearchInput

    def _run(self, query: str) -> str:
        return retrieve_faiss(query, k=5)


class WebSearchInput(BaseModel):
    query: str = Field(..., description="Search query for current public web information.")


class WebSearchTool(BaseTool):
    name: str = "Public Web Search"
    description: str = (
        "Searches the public web for current information. "
        "Prefer official government, university, institutional, or first-party sources "
        "for high-impact requirements."
    )
    args_schema: Type[BaseModel] = WebSearchInput

    def _run(self, query: str) -> str:
        try:
            results = DDGS().text(query, max_results=6)
            if not results:
                return "No web results found."

            formatted = []
            for item in results:
                title = item.get("title", "Untitled")
                url = item.get("href", "")
                body = item.get("body", "")
                formatted.append(
                    f"TITLE: {title}\nURL: {url}\nSNIPPET: {body}"
                )
            return "\n\n".join(formatted)
        except Exception as exc:
            return f"Web search unavailable: {exc}"


rag_tool = RAGSearchTool()
web_tool = WebSearchTool()


# ============================================================
# CrewAI setup
# ============================================================

def get_llm():
    if not GROQ_API_KEY:
        return None

    return LLM(
        model=f"groq/{MODEL_NAME}",
        api_key=GROQ_API_KEY,
        temperature=0.2,
        max_tokens=5000,
    )


def make_agents():
    llm = get_llm()
    if llm is None:
        return None

    supervisor = Agent(
        role="Life Event Supervisor",
        goal=(
            "Coordinate a team of specialized agents to turn a user's complex life "
            "transition into a realistic, dependency-aware, evidence-conscious plan."
        ),
        backstory=(
            "You are an experienced program manager who coordinates specialists. "
            "You do not invent facts. You distinguish evidence, assumptions, and "
            "user-specific decisions and keep the human in control."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    goal_agent = Agent(
        role="Life Goal Analyst",
        goal="Extract the user's life event, constraints, timeline, people, location, and missing information.",
        backstory="You convert natural language goals into structured planning requirements.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    research_agent = Agent(
        role="Research and Evidence Specialist",
        goal=(
            "Find relevant information from uploaded documents and public web sources. "
            "Prefer official first-party sources and clearly distinguish verified facts "
            "from assumptions."
        ),
        backstory=(
            "You are a careful research analyst. For immigration, financial, education, "
            "legal, health, or other high-impact matters, you prioritize authoritative sources."
        ),
        tools=[rag_tool, web_tool],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    planning_agent = Agent(
        role="Journey Planning Specialist",
        goal="Break the life event into concrete tasks and organize them into phases.",
        backstory="You specialize in turning vague goals into executable project plans.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    dependency_agent = Agent(
        role="Dependency Analyst",
        goal="Identify task dependencies, blockers, critical paths, and sequencing constraints.",
        backstory="You think in graphs and project dependencies and look for hidden blockers.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    risk_agent = Agent(
        role="Risk and Budget Analyst",
        goal="Identify risks, assumptions, cost categories, timing risks, and mitigation actions.",
        backstory="You stress-test plans and look for failure points before they become problems.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    critic_agent = Agent(
        role="Devil's Advocate and Quality Reviewer",
        goal=(
            "Challenge the proposed journey. Find missing tasks, weak evidence, "
            "unrealistic assumptions, contradictory dates, and dangerous overconfidence."
        ),
        backstory="You are deliberately skeptical. Your job is to make the plan safer and more robust.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    replanner_agent = Agent(
        role="Adaptive Replanning Specialist",
        goal=(
            "Produce a coherent final plan that incorporates dependencies, risks, "
            "evidence, critique, and the user's constraints."
        ),
        backstory=(
            "You specialize in adapting plans when dates, requirements, costs, or "
            "other assumptions change."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    return {
        "supervisor": supervisor,
        "goal": goal_agent,
        "research": research_agent,
        "planning": planning_agent,
        "dependency": dependency_agent,
        "risk": risk_agent,
        "critic": critic_agent,
        "replanner": replanner_agent,
    }


# ============================================================
# Crew execution
# ============================================================

def run_life_event(goal: str, budget: str, date: str, location: str, travelers: str):
    agents = make_agents()
    if not agents:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to Streamlit Secrets or the environment."
        )

    context = f"""
USER LIFE EVENT:
{goal}

USER CONSTRAINTS:
- Budget: {budget or "Not specified"}
- Target date: {date or "Not specified"}
- Current location: {location or "Not specified"}
- Travelers: {travelers or "Not specified"}

IMPORTANT:
- Do not invent legal, immigration, financial, university, or other high-impact requirements.
- Use web research and uploaded documents for factual claims.
- Mark assumptions clearly.
- The human remains the final decision-maker.
"""

    goal_task = Task(
        description=f"""
Analyze the following life event and return a structured understanding.

{context}

Return:
1. Event type
2. Origin and destination
3. Main objective
4. Target date
5. People involved
6. Explicit constraints
7. Missing information
8. Top planning priorities
""",
        expected_output="A structured life-event analysis.",
        agent=agents["goal"],
    )

    research_task = Task(
        description=f"""
Research the information needed to plan this life event.

{context}

Use:
- FAISS Document Search for uploaded user documents.
- Public Web Search for current public information.

For every important factual claim, provide:
- claim
- source/title
- URL if available
- whether the source is official/first-party
- confidence

Do not treat search snippets as guaranteed truth. Highlight information the user should verify.
""",
        expected_output="An evidence-aware research brief with sources.",
        agent=agents["research"],
        context=[goal_task],
    )

    planning_task = Task(
        description=f"""
Create a practical life journey for the user.

{context}

Produce 5-8 phases. For each phase, list concrete tasks with:
- task name
- purpose
- suggested timing
- priority
- completion criteria

Do not claim a task is legally mandatory unless supported by evidence.
""",
        expected_output="A phased task plan.",
        agent=agents["planning"],
        context=[goal_task, research_task],
    )

    dependency_task = Task(
        description=f"""
Analyze the proposed plan and create a dependency map.

{context}

For each important dependency, explain:
A -> B
because...

Identify:
- blockers
- prerequisites
- critical path
- tasks that can run in parallel
- tasks that should NOT be completed too early

Return a concise dependency graph in text.
""",
        expected_output="A dependency and critical-path analysis.",
        agent=agents["dependency"],
        context=[goal_task, research_task, planning_task],
    )

    risk_task = Task(
        description=f"""
Stress-test the life journey.

{context}

Identify at least:
- timing risks
- document/information risks
- financial risks
- dependency risks
- uncertainty risks

For each:
- risk
- likelihood: low/medium/high
- impact: low/medium/high
- affected tasks
- mitigation

Also provide a budget framework without fabricating exact prices.
""",
        expected_output="A risk register and budget framework.",
        agent=agents["risk"],
        context=[planning_task, dependency_task, research_task],
    )

    critic_task = Task(
        description=f"""
Act as a strict devil's advocate.

{context}

Review the goal, evidence, plan, dependencies, and risks.

Find:
1. Missing tasks
2. Unsupported factual claims
3. Unrealistic assumptions
4. Potential conflicts
5. Tasks that should be verified
6. One or more alternative approaches

Do not merely rewrite the plan. Challenge it.
""",
        expected_output="A critical review with actionable corrections.",
        agent=agents["critic"],
        context=[goal_task, research_task, planning_task, dependency_task, risk_task],
    )

    final_task = Task(
        description=f"""
Produce the final Life Event Navigator plan.

{context}

Combine all previous work into a clear user-facing result.

Use EXACTLY these sections:

# Life Event Summary
# Journey Phases
# Critical Dependencies
# Next 5 Actions
# Timeline
# Budget Framework
# Risks and Mitigations
# Evidence and Sources
# What You Should Verify
# If Something Changes

Rules:
- Do not present guesses as facts.
- Clearly label assumptions.
- Prefer official sources for high-impact requirements.
- Include URLs from research where available.
- Keep the human in control of consequential decisions.
- The "If Something Changes" section must explain how the plan should be recalculated if a major date or requirement changes.
""",
        expected_output="A complete, evidence-aware, dependency-aware life journey.",
        agent=agents["replanner"],
        context=[
            goal_task,
            research_task,
            planning_task,
            dependency_task,
            risk_task,
            critic_task,
        ],
    )

    crew = Crew(
        agents=list(agents.values()),
        tasks=[
            goal_task,
            research_task,
            planning_task,
            dependency_task,
            risk_task,
            critic_task,
            final_task,
        ],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    return result


# ============================================================
# Replanning
# ============================================================

def run_replan(existing_plan: str, change: str):
    agents = make_agents()
    if not agents:
        raise RuntimeError("GROQ_API_KEY is missing.")

    task = Task(
        description=f"""
You are the adaptive replanning agent.

EXISTING PLAN:
{existing_plan}

USER CHANGE:
{change}

Determine:
1. What changed?
2. Which assumptions are affected?
3. Which tasks are affected?
4. Which dependencies are affected?
5. What dates or sequencing should change?
6. What risks have increased or decreased?
7. What actions should the user take next?

Return:
# Change Impact
# Affected Tasks
# Updated Dependencies
# Updated Timeline
# Updated Risks
# New Next Actions
# What the User Should Verify

Do not invent current legal or regulatory requirements.
""",
        expected_output="An updated plan impact analysis and revised next actions.",
        agent=agents["replanner"],
    )

    crew = Crew(
        agents=[agents["replanner"]],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )
    return crew.kickoff()


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("⚙️ Configuration")

    if GROQ_API_KEY:
        st.success("Groq API key detected")
    else:
        st.error("Groq API key missing")

    st.caption(
        "Add GROQ_API_KEY in Streamlit Secrets. Never commit your API key to GitHub."
    )

    st.divider()

    st.header("📄 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload supporting PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Admission letters, university guides, checklists, official documents, etc.",
    )

    if st.button("📥 Index PDFs", use_container_width=True):
        if not uploaded_files:
            st.warning("Upload at least one PDF first.")
        else:
            chunks = []
            sources = []

            for uploaded in uploaded_files:
                try:
                    reader = PdfReader(uploaded)
                    text = "\n".join(
                        page.extract_text() or "" for page in reader.pages
                    )
                    file_chunks = chunk_text(text)

                    chunks.extend(file_chunks)
                    sources.extend([uploaded.name] * len(file_chunks))
                except Exception as exc:
                    st.error(f"Could not read {uploaded.name}: {exc}")

            st.session_state["knowledge_chunks"] = chunks
            st.session_state["knowledge_sources"] = sources

            if chunks:
                st.success(f"Indexed {len(chunks)} chunks.")
            else:
                st.warning("No extractable text was found.")

    if st.session_state["knowledge_chunks"]:
        st.info(
            f"{len(st.session_state['knowledge_chunks'])} document chunks "
            "available to the agents."
        )

    st.divider()

    st.header("🤖 Agent Team")

    agents_display = [
        "🎯 Goal Analyst",
        "🔎 Research & Evidence",
        "🗺️ Journey Planner",
        "🔗 Dependency Analyst",
        "⚠️ Risk & Budget",
        "🧨 Devil's Advocate",
        "♻️ Adaptive Replanner",
    ]

    for agent_name in agents_display:
        st.markdown(f"- {agent_name}")


# ============================================================
# Main input
# ============================================================

st.header("Create Your Life Journey")

goal = st.text_area(
    "What major life event are you planning?",
    value=(
        "I have been accepted into a master's degree program abroad and "
        "need to plan my move, documents, finances, accommodation, travel, "
        "and university arrival."
    ),
    height=130,
)

col1, col2 = st.columns(2)

with col1:
    location = st.text_input("Current location", placeholder="e.g. Pakistan")
    date = st.text_input(
        "Target date",
        placeholder="e.g. September 2027",
    )

with col2:
    budget = st.text_input(
        "Approximate budget",
        placeholder="e.g. USD 20,000",
    )
    travelers = st.text_input(
        "Who is traveling?",
        placeholder="e.g. Me alone / family of 4",
    )

create = st.button(
    "🚀 Create My Life Journey",
    type="primary",
    use_container_width=True,
)

if create:
    if not GROQ_API_KEY:
        st.error(
            "GROQ_API_KEY is missing. Add it under Streamlit Cloud → "
            "App settings → Secrets."
        )
    elif not goal.strip():
        st.warning("Please describe the life event first.")
    else:
        with st.status("🤖 AI agents are building your journey...", expanded=True) as status:
            st.write("🎯 Goal Analyst: understanding your life event")
            st.write("🔎 Research Agent: gathering evidence")
            st.write("🗺️ Planning Agent: creating tasks")
            st.write("🔗 Dependency Agent: finding blockers")
            st.write("⚠️ Risk Agent: stress-testing the journey")
            st.write("🧨 Critic Agent: challenging the plan")
            st.write("♻️ Replanner: assembling the final journey")

            try:
                result = run_life_event(
                    goal=goal,
                    budget=budget,
                    date=date,
                    location=location,
                    travelers=travelers,
                )

                final_text = result.raw if hasattr(result, "raw") else str(result)
                st.session_state["plan"] = final_text
                st.session_state["last_run"] = {
                    "goal": goal,
                    "budget": budget,
                    "date": date,
                    "location": location,
                    "travelers": travelers,
                }

                status.update(
                    label="✅ Life journey created",
                    state="complete",
                    expanded=False,
                )

            except Exception as exc:
                status.update(
                    label="❌ Journey generation failed",
                    state="error",
                    expanded=True,
                )
                st.exception(exc)


# ============================================================
# Display plan
# ============================================================

if st.session_state["plan"]:
    st.divider()
    st.header("🧭 Your Life Journey")

    st.markdown(st.session_state["plan"])

    st.divider()

    st.header("♻️ Something Changed?")

    change = st.text_area(
        "Describe a change and the AI will calculate its impact.",
        placeholder=(
            "Example: My visa appointment was delayed by four weeks. "
            "What does this change affect?"
        ),
        height=100,
    )

    if st.button("🔄 Recalculate My Journey", use_container_width=True):
        if not change.strip():
            st.warning("Describe the change first.")
        else:
            with st.spinner("Agents are analyzing the impact..."):
                try:
                    updated = run_replan(
                        existing_plan=st.session_state["plan"],
                        change=change,
                    )
                    updated_text = updated.raw if hasattr(updated, "raw") else str(updated)

                    st.session_state["plan"] = (
                        st.session_state["plan"]
                        + "\n\n---\n\n# ♻️ Latest Replanning Update\n\n"
                        + updated_text
                    )

                    st.success("Journey recalculated.")
                    st.rerun()

                except Exception as exc:
                    st.exception(exc)


# ============================================================
# Footer
# ============================================================

st.divider()
st.caption(
    "Life Event Navigator is an AI planning prototype. "
    "Verify high-impact requirements with authoritative sources before acting."
)
