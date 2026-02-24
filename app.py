# app.py — Main Streamlit application for the Multi-Agent SIR Disease Simulator

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Make the src/ package importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agents.analyzer_agent import AnalyzerAgent
from agents.rag_agent import RAGAgent
from agents.reporter_agent import ReporterAgent
from sir_sim import main as run_simulation
from utils.analysis_tools import (
    calculate_average_total_infected,
    calculate_final_state_distribution,
    calculate_infection_decline_rate,
    calculate_peak_infection,
    calculate_peak_infection_std,
    calculate_time_to_half_infected,
    plot_state_dynamics,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SIR Disease Simulator",
    page_icon="🦠",
    layout="wide",
)

st.title("🦠 Multi-Agent SIR Infectious Disease Simulator")
st.caption(
    "A multi-agent AI system for simulating and analyzing epidemic spread "
    "using stochastic SIR models."
)

# ── Groq API key ──────────────────────────────────────────────────────────────
# Read from Streamlit Secrets first, fall back to nothing (user can paste in UI).
_GROQ_KEY_FROM_SECRETS = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""

# ── Session state ─────────────────────────────────────────────────────────────
if "sim_data" not in st.session_state:
    st.session_state.sim_data = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, "logs", "all_agent_logs.csv")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# ── Sidebar — Groq API key ────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙ Configuration")
    st.markdown(
        "The **simulation** and **analysis** tabs work without any API key. "
        "The **AI Report** and **Learn / Q&A** tabs need a [Groq API key](https://console.groq.com) "
        "(free tier available)."
    )
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=_GROQ_KEY_FROM_SECRETS,
        placeholder="gsk_...",
        help="Get a free key at console.groq.com",
    )
    groq_model = st.selectbox(
        "Groq Model",
        ["llama-3.3-70b-versatile", "llama3-8b-8192", "mixtral-8x7b-32768"],
    )
    st.divider()
    st.caption("Source: [GitHub](https://github.com)")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["▶ Run Simulation", "📊 Analyze Results", "🤖 AI Report", "📚 Learn / Q&A"]
)

# =============================================================================
# TAB 1 — RUN SIMULATION
# =============================================================================
with tab1:
    st.subheader("Configure and Run the Simulation")
    st.info(
        "Adjust the parameters below and click **Run Simulation**. "
        "No API key required — the simulation runs entirely in Python.",
        icon="ℹ️",
    )

    col1, col2 = st.columns(2)
    with col1:
        num_agents = st.slider("Number of Agents", 100, 5000, 1000, step=100)
        num_steps = st.slider("Simulation Steps (days)", 10, 100, 28)
        num_runs = st.slider("Monte Carlo Runs", 1, 200, 50)
        num_contacts = st.slider("Group Contacts per Step", 2, 50, 10)
    with col2:
        infection_prob = st.slider(
            "Infection Probability per Contact", 0.01, 1.0, 0.3, step=0.01
        )
        recovery_prob = st.slider(
            "Recovery Probability per Step", 0.01, 1.0, 0.1, step=0.01
        )
        infection_duration = st.slider("Infection Duration (steps)", 1, 30, 3)
        seed = st.number_input("Random Seed", value=42, step=1)

    if st.button("▶ Run Simulation", type="primary", use_container_width=True):
        params = {
            "seed": int(seed),
            "num_runs": num_runs,
            "num_agents": num_agents,
            "num_steps": num_steps,
            "num_contacts": num_contacts,
            "infection_prob": infection_prob,
            "infection_duration": infection_duration,
            "recovery_prob": recovery_prob,
        }

        progress_bar = st.progress(0, text="Starting simulation…")

        def _update_progress(completed, total):
            progress_bar.progress(completed / total, text=f"Run {completed}/{total}")

        with st.spinner(f"Running {num_runs} simulation runs…"):
            df = run_simulation(
                params, log_path=LOG_PATH, progress_callback=_update_progress
            )

        progress_bar.empty()
        st.session_state.sim_data = df
        st.success(
            f"Done! {num_runs} runs × {num_agents:,} agents × {num_steps} steps."
        )

    if st.session_state.sim_data is not None:
        df = st.session_state.sim_data
        st.subheader("SIR State Dynamics")
        fig = plot_state_dynamics(df)
        st.pyplot(fig)
        plt.close(fig)

        with st.expander("View raw simulation data (first 1,000 rows)"):
            st.dataframe(df.head(1000), use_container_width=True)
            st.download_button(
                "Download full CSV",
                df.to_csv(index=False).encode(),
                "sir_simulation.csv",
                "text/csv",
            )

# =============================================================================
# TAB 2 — ANALYZE RESULTS
# =============================================================================
with tab2:
    st.subheader("Analysis Dashboard")

    if st.session_state.sim_data is None:
        st.info("Run the simulation first (Tab 1) to enable analysis.", icon="ℹ️")
    else:
        df = st.session_state.sim_data

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            peak, peak_step = calculate_peak_infection(df)
            std = calculate_peak_infection_std(df)
            st.metric("Peak Infected Agents", f"{peak:,}")
            st.metric("Peak Infection Day", f"Day {peak_step}")
            st.metric("Peak Std Dev", f"±{std:,}")

        with col_b:
            avg_total = calculate_average_total_infected(df)
            decline = calculate_infection_decline_rate(df)
            st.metric("Avg Total Ever Infected", f"{avg_total:,}")
            st.metric("Post-Peak Decline Rate", f"{decline}% / step")

        with col_c:
            half_step = calculate_time_to_half_infected(df)
            final_dist = calculate_final_state_distribution(df)
            st.metric("Steps to 50% Infected", f"Day {half_step}")
            st.metric("Final Susceptible", f"{final_dist.get('Susceptible', 0):,}")
            st.metric("Final Recovered", f"{final_dist.get('Recovered', 0):,}")
            st.metric("Final Infected", f"{final_dist.get('Infected', 0):,}")

        st.divider()
        st.subheader("Final State Distribution (avg across runs)")
        dist_df = pd.DataFrame(
            list(final_dist.items()), columns=["State", "Avg Agents"]
        )
        st.bar_chart(dist_df.set_index("State"))

# =============================================================================
# TAB 3 — AI REPORT
# =============================================================================
with tab3:
    st.subheader("AI-Generated Analysis Report")

    if st.session_state.sim_data is None:
        st.info("Run the simulation first (Tab 1) to enable AI reporting.", icon="ℹ️")
    elif not groq_api_key:
        st.warning(
            "Add your Groq API key in the sidebar to use AI features.", icon="🔑"
        )
    else:
        st.markdown(
            "Ask the **Reporter Agent** to explain your simulation results in plain language."
        )

        preset_questions = [
            "Summarize the key findings from this simulation.",
            "What do the peak infection statistics tell us about this outbreak?",
            "How quickly did the infection spread through the population?",
            "What does the final state distribution tell us about herd immunity?",
            "How does the recovery rate compare to the infection rate?",
        ]

        question_choice = st.selectbox(
            "Choose a preset question or type your own:",
            ["(Type your own…)"] + preset_questions,
            key="reporter_question_select",
        )
        if question_choice == "(Type your own…)":
            report_question = st.text_input(
                "Your question:",
                placeholder="e.g. What was the peak infection rate?",
                key="reporter_custom_q",
            )
        else:
            report_question = question_choice

        if st.button("Generate Report", type="primary", key="generate_report"):
            if not report_question:
                st.warning("Please enter or select a question.")
            else:
                analyzer = AnalyzerAgent(state_data=st.session_state.sim_data)
                analysis_results = analyzer.analyze(report_question)
                reporter = ReporterAgent(
                    backend="groq", model=groq_model, api_key=groq_api_key
                )

                with st.spinner("Generating report…"):
                    try:
                        response = reporter.report(report_question, analysis_results)
                        st.markdown("### Reporter Agent Response")
                        st.markdown(response)
                        if analysis_results:
                            with st.expander("Raw analysis data"):
                                st.json(analysis_results)
                    except Exception as e:
                        st.error(f"LLM error: {e}")

# =============================================================================
# TAB 4 — LEARN / Q&A
# =============================================================================
with tab4:
    st.subheader("Learn About SIR Models")
    st.markdown(
        "Ask the **RAG Agent** questions about the SIR model, its assumptions, "
        "and infectious disease spread. It searches the built-in knowledge base "
        "and generates an answer using Groq."
    )

    if not groq_api_key:
        st.warning(
            "Add your Groq API key in the sidebar to use AI features.", icon="🔑"
        )
    else:
        sample_questions = [
            "What are the assumptions of this SIR model?",
            "Who developed the SIR model and when?",
            "What is R₀ and why does it matter?",
            "How was the SIR model used during COVID-19?",
            "Why is stochastic modeling important in public health?",
            "How do SIR models help decision makers?",
        ]

        learn_choice = st.selectbox(
            "Sample questions:",
            ["(Type your own…)"] + sample_questions,
            key="rag_question_select",
        )
        if learn_choice == "(Type your own…)":
            learn_question = st.text_input(
                "Your question:",
                placeholder="e.g. What is the recovery rate?",
                key="rag_custom_q",
            )
        else:
            learn_question = learn_choice

        if st.button("Ask RAG Agent", type="primary", key="ask_rag"):
            if not learn_question:
                st.warning("Please enter or select a question.")
            else:
                rag = RAGAgent(
                    backend="groq", model=groq_model, api_key=groq_api_key
                )
                with st.spinner("Searching knowledge base and generating answer…"):
                    try:
                        answer = rag.answer(learn_question)
                        st.markdown("### RAG Agent Response")
                        st.markdown(answer)
                    except Exception as e:
                        st.error(f"LLM error: {e}")

    st.divider()
    with st.expander("📖 View full SIR model knowledge base"):
        kb_path = os.path.join(
            BASE_DIR, "src", "knowledge", "sir_model_information.md"
        )
        if os.path.exists(kb_path):
            with open(kb_path) as f:
                st.markdown(f.read())
        else:
            st.warning("Knowledge base file not found.")
