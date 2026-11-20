"""Streamlit interface for PolicyRAG with high-contrast visibility, token streaming, and User/Dev modes."""

from __future__ import annotations

import html
import re
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

from scripts.create_mock_pdfs import build_policy_pdf
from src.answering import (
    RAG_MODE,
    STRICT_MODE,
    AnswerResult,
    answer_rag_synthesis,
    answer_strict_retrieval,
)
from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COMPANIES_DIR,
    COMPANIES_INDEX_DIR,
    DEFAULT_RELEVANCE_THRESHOLD,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    MOCK_POLICY_DIR,
)
from src.indexing import (
    build_faiss_index,
    index_exists,
    list_policy_pdfs,
    read_index_manifest,
)
from src.mock_docs import MOCK_POLICIES
from src.ollama_client import list_ollama_models, select_ollama_model, stream_with_ollama
from src.utils import truncate_text


EXAMPLE_QUESTIONS = [
    "What documents are required for corporate onboarding?",
    "When is enhanced due diligence required?",
    "What are the rules for confidential client data?",
    "What should happen during a SEV1 incident?",
    "Can the AI assistant answer using information outside approved documents?",
]

DEFAULT_COMPANY = "Default Enterprise"
USER_MODE = "User Mode"
DEV_MODE = "Developer Mode"


def sanitize_company_name(name: str) -> str:
    clean = re.sub(r'[\\/*?:"<>|]', "", name).strip()
    return clean or "Unnamed Company"


def list_companies() -> List[str]:
    COMPANIES_DIR.mkdir(parents=True, exist_ok=True)
    dirs = [d.name for d in sorted(COMPANIES_DIR.iterdir()) if d.is_dir()]
    if not dirs:
        init_default_company()
        dirs = [DEFAULT_COMPANY]
    return dirs


def get_company_pdf_dir(company: str) -> Path:
    target = COMPANIES_DIR / company
    target.mkdir(parents=True, exist_ok=True)
    return target


def get_company_index_dir(company: str) -> Path:
    target = COMPANIES_INDEX_DIR / company
    target.mkdir(parents=True, exist_ok=True)
    return target


def init_default_company() -> None:
    default_dir = get_company_pdf_dir(DEFAULT_COMPANY)
    if not any(default_dir.glob("*.pdf")):
        if MOCK_POLICY_DIR.exists() and any(MOCK_POLICY_DIR.glob("*.pdf")):
            for p in MOCK_POLICY_DIR.glob("*.pdf"):
                shutil.copy(p, default_dir / p.name)
        else:
            for policy in MOCK_POLICIES:
                build_policy_pdf(policy, default_dir)


@st.cache_data(ttl=5)
def cached_company_manifest(company: str) -> Optional[Dict[str, object]]:
    return read_index_manifest(get_company_index_dir(company))


@st.cache_data(ttl=10)
def cached_ollama_models() -> List[str]:
    return list_ollama_models()


def clear_app_caches() -> None:
    cached_company_manifest.clear()
    cached_ollama_models.clear()


def render_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-base: #000000;
            --bg-surface: rgba(255, 255, 255, 0.05);
            --bg-surface-hover: rgba(255, 255, 255, 0.09);
            --border-subtle: rgba(255, 255, 255, 0.16);
            --border-hover: rgba(255, 255, 255, 0.35);
            --text-pure: #ffffff;
            --text-primary: #ffffff;
            --text-secondary: #c5c5d0;
            --text-muted: #8e8e98;
            --radius-sm: 8px;
            --radius-md: 14px;
            --radius-lg: 20px;
            --radius-pill: 9999px;
        }

        .stApp {
            background-color: var(--bg-base) !important;
            background-image: none !important;
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Helvetica Neue", sans-serif !important;
            color: #ffffff !important;
        }

        .main .block-container {
            max-width: 1040px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        section[data-testid="stSidebar"] {
            background: #080808 !important;
            border-right: 1px solid var(--border-subtle) !important;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] p {
            color: #ffffff !important;
        }

        /* High-contrast Segmented Control & Action Buttons */
        div[data-testid="stButton"] button[kind="primary"],
        section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] {
            background-color: #ffffff !important;
            background: #ffffff !important;
            border: 1px solid #ffffff !important;
            border-radius: var(--radius-pill) !important;
            padding: 0.55rem 1.4rem !important;
            font-weight: 700 !important;
            font-size: 0.92rem !important;
            letter-spacing: -0.01em !important;
            box-shadow: 0 2px 10px rgba(255, 255, 255, 0.25) !important;
        }

        div[data-testid="stButton"] button[kind="primary"] *,
        section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] * {
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
            background: transparent !important;
            background-color: transparent !important;
            font-weight: 700 !important;
        }

        div[data-testid="stButton"] button[kind="secondary"],
        section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="secondary"] {
            background: rgba(255, 255, 255, 0.06) !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.22) !important;
            border-radius: var(--radius-pill) !important;
            padding: 0.55rem 1.4rem !important;
            font-size: 0.88rem !important;
            font-weight: 500 !important;
            transition: all 0.15s ease !important;
        }

        div[data-testid="stButton"] button[kind="secondary"] *,
        section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="secondary"] * {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }

        div[data-testid="stButton"] button[kind="secondary"]:hover,
        section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="secondary"]:hover {
            background: rgba(255, 255, 255, 0.14) !important;
            border-color: rgba(255, 255, 255, 0.45) !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }

        /* Hero Header */
        .hero-wrap {
            padding: 0.5rem 0 1.6rem 0;
            border-bottom: 1px solid var(--border-subtle);
            margin-bottom: 1.6rem;
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
        }

        .hero-title {
            font-size: 2.15rem;
            font-weight: 700;
            letter-spacing: -0.04em;
            color: #ffffff;
            margin: 0 0 0.35rem 0;
            line-height: 1.15;
        }

        .hero-subtitle {
            font-size: 0.96rem;
            color: #c5c5d0;
            margin: 0;
            font-weight: 400;
            letter-spacing: -0.01em;
        }

        .mode-pill-badge {
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            padding: 0.25rem 0.75rem;
            border-radius: var(--radius-pill);
            background: rgba(255, 255, 255, 0.15);
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: #ffffff;
        }

        /* High-Contrast Answer Card */
        .answer-card {
            background: #111114 !important;
            border: 1px solid rgba(255, 255, 255, 0.28) !important;
            border-radius: var(--radius-lg) !important;
            padding: 1.8rem 2rem !important;
            margin: 1.4rem 0 !important;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.75) !important;
        }

        .badge-pill {
            display: inline-block !important;
            padding: 0.4rem 1rem !important;
            border-radius: var(--radius-pill) !important;
            font-size: 0.78rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.06em !important;
            text-transform: uppercase !important;
            background: rgba(255, 255, 255, 0.18) !important;
            border: 1px solid rgba(255, 255, 255, 0.45) !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            margin-bottom: 1.2rem !important;
        }

        .answer-body {
            font-size: 1.08rem !important;
            line-height: 1.8 !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            font-weight: 400 !important;
        }

        .answer-body, .answer-body * {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }

        .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li, .stMarkdown div {
            color: #ffffff !important;
        }

        /* High-Contrast Source Chips */
        .citation-wrap {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
            margin-top: 0.6rem;
        }

        .citation-chip {
            display: inline-block !important;
            background: #18181c !important;
            border: 1px solid rgba(255, 255, 255, 0.35) !important;
            border-radius: var(--radius-pill) !important;
            padding: 0.45rem 1.05rem !important;
            font-size: 0.85rem !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            font-weight: 500 !important;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4) !important;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        }

        .excerpt-block {
            background: rgba(255, 255, 255, 0.03);
            border-left: 3px solid #ffffff;
            border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
            padding: 0.95rem 1.2rem;
            margin-bottom: 0.9rem;
        }

        .excerpt-text {
            font-size: 0.98rem;
            line-height: 1.6;
            color: #ffffff;
        }

        .excerpt-meta {
            font-size: 0.82rem;
            color: #c5c5d0;
            margin-top: 0.5rem;
            font-weight: 500;
        }

        .section-label {
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #c5c5d0;
            margin: 1.8rem 0 0.8rem 0;
        }

        div[data-testid="stTextArea"] textarea {
            background: rgba(255, 255, 255, 0.04) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: var(--radius-md) !important;
            color: #ffffff !important;
            font-size: 1.02rem !important;
            padding: 1.1rem 1.2rem !important;
            line-height: 1.5 !important;
        }

        div[data-testid="stTextArea"] textarea:focus {
            border-color: rgba(255, 255, 255, 0.6) !important;
            background: rgba(255, 255, 255, 0.07) !important;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.85rem;
            margin: 1.2rem 0;
        }

        @media (max-width: 768px) {
            .metrics-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }

        .metric-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 0.95rem 1.1rem;
        }

        .metric-label {
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #8e8e98;
            margin-bottom: 0.4rem;
        }

        .metric-val {
            font-size: 1.35rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            color: #ffffff;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .metric-sub {
            font-size: 0.76rem;
            color: #c5c5d0;
            margin-top: 0.2rem;
        }

        .doc-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.65rem 0.85rem;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-sm);
            margin-bottom: 0.45rem;
            font-size: 0.85rem;
        }

        div[data-testid="stExpander"] {
            background: var(--bg-surface) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: var(--radius-md) !important;
            margin-bottom: 0.65rem !important;
        }

        div[data-testid="stExpander"] summary {
            font-size: 0.88rem !important;
            font-weight: 600 !important;
            color: #ffffff !important;
            padding: 0.8rem 1rem !important;
        }

        div[data-testid="stTextInput"] div[data-baseweb="input"] {
            display: none;
        }
        div[data-testid="stTextInput"] {
            margin-bottom: -0.45rem;
        }
        header[data-testid="stHeader"] button[data-testid="stBaseButton-header"] {
            display: none !important;
        }
        div[data-testid="stToast"],
        div[data-testid="stNotification"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_session_state() -> None:
    if "question" not in st.session_state:
        st.session_state.question = EXAMPLE_QUESTIONS[0]
    if "selected_company" not in st.session_state:
        st.session_state.selected_company = DEFAULT_COMPANY
    if "view_mode" not in st.session_state:
        st.session_state.view_mode = USER_MODE
    if "answer_mode_choice" not in st.session_state:
        st.session_state.answer_mode_choice = RAG_MODE


def set_question(question: str) -> None:
    st.session_state.question = question


def render_segmented_mode_switch() -> str:
    st.sidebar.markdown(
        """
        <div style="font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:#8e8e98; margin-bottom:0.4rem;">
            Mode
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_u, col_d = st.sidebar.columns(2)
    current_mode = st.session_state.view_mode

    with col_u:
        is_user = current_mode == USER_MODE
        if st.button(
            "User",
            key="btn_mode_user",
            type="primary" if is_user else "secondary",
            use_container_width=True,
        ):
            if not is_user:
                st.session_state.view_mode = USER_MODE
                st.rerun()

    with col_d:
        is_dev = current_mode == DEV_MODE
        if st.button(
            "Developer",
            key="btn_mode_dev",
            type="primary" if is_dev else "secondary",
            use_container_width=True,
        ):
            if not is_dev:
                st.session_state.view_mode = DEV_MODE
                st.rerun()

    return st.session_state.view_mode


def render_sidebar() -> Dict[str, object]:
    view_mode = render_segmented_mode_switch()

    st.sidebar.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.12); margin:1rem 0;'>", unsafe_allow_html=True)

    st.sidebar.markdown(
        """
        <div style="font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:#8e8e98; margin-bottom:0.4rem;">
            Organization
        </div>
        """,
        unsafe_allow_html=True,
    )

    companies = list_companies()
    if st.session_state.selected_company not in companies:
        st.session_state.selected_company = companies[0]

    selected_company = st.sidebar.selectbox(
        "Active Company",
        companies,
        index=companies.index(st.session_state.selected_company),
        label_visibility="collapsed",
        help="Select the company workspace to search.",
    )
    st.session_state.selected_company = selected_company

    st.sidebar.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.12); margin:1rem 0;'>", unsafe_allow_html=True)
    st.sidebar.markdown(
        """
        <div style="font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:#8e8e98; margin-bottom:0.4rem;">
            Answer Mode
        </div>
        """,
        unsafe_allow_html=True,
    )

    mode = st.sidebar.radio(
        "Answer Mode",
        [RAG_MODE, STRICT_MODE],
        index=0 if st.session_state.get("answer_mode_choice") != STRICT_MODE else 1,
        label_visibility="collapsed",
        key="answer_mode_selection",
        help="RAG Synthesis uses Ollama to generate a synthesized response with citations. Strict Retrieval quotes exact policy excerpts with zero hallucination.",
    )
    st.session_state.answer_mode_choice = mode

    available_models = cached_ollama_models()

    if view_mode == DEV_MODE:
        st.sidebar.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.12); margin:1rem 0;'>", unsafe_allow_html=True)
        st.sidebar.markdown(
            """
            <div style="font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:#8e8e98; margin-bottom:0.4rem;">
                Tuning & Speed
            </div>
            """,
            unsafe_allow_html=True,
        )

        model_options = ["Auto fallback"] + available_models
        model_disabled = mode == STRICT_MODE
        selected_model_label = st.sidebar.selectbox(
            "Ollama Model",
            model_options,
            disabled=model_disabled,
            help="Select local LLM. llama3.2:1b is optimized for fast CPU speed.",
        )
        selected_model = None if selected_model_label == "Auto fallback" else selected_model_label

        top_k = st.sidebar.slider(
            "Retrieved Chunks (Top-K)",
            min_value=1,
            max_value=8,
            value=DEFAULT_TOP_K,
            step=1,
            help="Number of chunks retrieved from the FAISS vector index.",
        )
        threshold = st.sidebar.slider(
            "Relevance Threshold",
            min_value=0.0,
            max_value=0.9,
            value=DEFAULT_RELEVANCE_THRESHOLD,
            step=0.05,
            help="Refusal boundary.",
        )

        st.sidebar.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.12); margin:1rem 0;'>", unsafe_allow_html=True)
        rebuild_clicked = st.sidebar.button(f"Rebuild Index ({selected_company})", use_container_width=True)
        if rebuild_clicked:
            pdf_dir = get_company_pdf_dir(selected_company)
            index_dir = get_company_index_dir(selected_company)
            with st.spinner(f"Rebuilding vector index for {selected_company}..."):
                result = build_faiss_index(pdf_dir=pdf_dir, index_dir=index_dir, force=True)
            clear_app_caches()
            st.sidebar.success(f"Indexed {result.chunk_count} chunks from {result.pdf_count} PDFs.")

    else:
        # User Mode defaults
        selected_model = None
        top_k = DEFAULT_TOP_K
        threshold = DEFAULT_RELEVANCE_THRESHOLD

    st.sidebar.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.12); margin:1rem 0;'>", unsafe_allow_html=True)
    st.sidebar.markdown(
        """
        <div style="font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:#8e8e98; margin-bottom:0.5rem;">
            Sample Questions
        </div>
        """,
        unsafe_allow_html=True,
    )
    for idx, example in enumerate(EXAMPLE_QUESTIONS, start=1):
        st.sidebar.button(
            example,
            key=f"example_{idx}",
            use_container_width=True,
            on_click=set_question,
            args=(example,),
        )

    return {
        "view_mode": view_mode,
        "company": selected_company,
        "mode": mode,
        "selected_model": selected_model,
        "available_models": available_models,
        "top_k": top_k,
        "threshold": threshold,
    }


def render_hero(company: str, view_mode: str) -> None:
    if view_mode == USER_MODE:
        st.markdown(
            f"""
            <div class="hero-wrap">
                <div>
                    <h1 class="hero-title">Policy Intelligence</h1>
                    <p class="hero-subtitle">Search verified policies for <strong style="color:#ffffff;">{html.escape(company)}</strong>.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="hero-wrap">
                <div>
                    <h1 class="hero-title">PolicyRAG <span class="mode-pill-badge">DEVELOPER</span></h1>
                    <p class="hero-subtitle">Vector telemetry, chunks inspection, and model tuning for <strong style="color:#ffffff;">{html.escape(company)}</strong>.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_status_panel(controls: Dict[str, object]) -> None:
    if controls["view_mode"] != DEV_MODE:
        company = str(controls["company"])
        company_index_dir = get_company_index_dir(company)
        if not index_exists(company_index_dir):
            st.markdown(
                f"""
                <div style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.2); border-radius:8px; padding:0.85rem 1.1rem; color:#ffffff; font-size:0.9rem; margin:1rem 0;">
                    No documents indexed yet for <strong>{html.escape(company)}</strong>. Upload a policy PDF below to begin.
                </div>
                """,
                unsafe_allow_html=True,
            )
        return

    company = str(controls["company"])
    manifest = cached_company_manifest(company)
    company_pdf_dir = get_company_pdf_dir(company)
    pdf_count = len(list_policy_pdfs(company_pdf_dir))
    chunk_count = manifest.get("chunk_count", 0) if manifest else 0
    active_selection = select_ollama_model(preferred_model=controls["selected_model"])
    active_model = active_selection.selected_model or "Unavailable"
    embedding_name = EMBEDDING_MODEL_NAME.split("/")[-1]

    st.markdown(
        f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Organization</div>
                <div class="metric-val" title="{html.escape(company)}">{html.escape(company)}</div>
                <div class="metric-sub">{pdf_count} active PDF(s)</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Vector Store</div>
                <div class="metric-val">{chunk_count}</div>
                <div class="metric-sub">Size {CHUNK_SIZE}, overlap {CHUNK_OVERLAP}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Embeddings</div>
                <div class="metric-val" title="{embedding_name}">{embedding_name}</div>
                <div class="metric-sub">384-d normalized CPU</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Language Model</div>
                <div class="metric-val" title="{active_model}">{active_model}</div>
                <div class="metric-sub">{'Connected (High-Speed)' if active_selection.ollama_available else 'Offline'}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_document_management(active_company: str, view_mode: str) -> None:
    if view_mode == USER_MODE:
        with st.expander(f"Upload Policies to {active_company}", expanded=False):
            st.markdown(
                f"""
                <div style="font-size:0.85rem; color:#c5c5d0; margin-bottom:0.75rem;">
                    Upload one or more PDF files to update <strong>{html.escape(active_company)}</strong>'s knowledge base.
                </div>
                """,
                unsafe_allow_html=True,
            )
            uploaded_files = st.file_uploader(
                "Upload policy documents (PDF)",
                type=["pdf"],
                accept_multiple_files=True,
                label_visibility="collapsed",
                key="user_upload_pdfs",
            )
            if st.button("Upload & Index", type="primary", use_container_width=True):
                if not uploaded_files:
                    st.warning("Please choose at least one PDF file.")
                else:
                    pdf_dir = get_company_pdf_dir(active_company)
                    index_dir = get_company_index_dir(active_company)
                    with st.spinner("Indexing documents..."):
                        for uploaded in uploaded_files:
                            target = pdf_dir / uploaded.name
                            with open(target, "wb") as f:
                                f.write(uploaded.getbuffer())
                        build_faiss_index(pdf_dir=pdf_dir, index_dir=index_dir, force=True)
                        clear_app_caches()
                    st.success(f"Added {len(uploaded_files)} document(s) to {active_company}.")
                    st.rerun()

    else:
        with st.expander("Company Workspace & File Telemetry", expanded=False):
            tab_upload, tab_new_company, tab_files = st.tabs([
                "Upload Documents",
                "Add New Company",
                f"Active Files ({active_company})",
            ])

            with tab_upload:
                uploaded_files = st.file_uploader(
                    f"Upload PDFs to {active_company}",
                    type=["pdf"],
                    accept_multiple_files=True,
                    key=f"uploader_{active_company}",
                )

                replace_existing = st.checkbox(
                    f"Replace existing documents in {active_company}",
                    value=False,
                    key="replace_company_docs",
                )

                if st.button("Process & Index for this Company", type="primary", use_container_width=True):
                    if not uploaded_files:
                        st.warning("Please choose at least one PDF file.")
                    else:
                        pdf_dir = get_company_pdf_dir(active_company)
                        index_dir = get_company_index_dir(active_company)
                        with st.spinner(f"Indexing {len(uploaded_files)} PDF(s) for {active_company}..."):
                            if replace_existing:
                                for old in pdf_dir.glob("*.pdf"):
                                    old.unlink()

                            for uploaded in uploaded_files:
                                target = pdf_dir / uploaded.name
                                with open(target, "wb") as f:
                                    f.write(uploaded.getbuffer())

                            result = build_faiss_index(pdf_dir=pdf_dir, index_dir=index_dir, force=True)
                            clear_app_caches()

                        st.success(f"Indexed {len(uploaded_files)} document(s) for {active_company}. Total: {result.chunk_count} chunks.")
                        st.rerun()

            with tab_new_company:
                col_name, col_create = st.columns([2, 1])
                with col_name:
                    new_company_name = st.text_input(
                        "Company Name",
                        placeholder="e.g. Acme Corp, TechHealth, FinCore...",
                        label_visibility="collapsed",
                        key="input_new_company_name",
                    )
                with col_create:
                    btn_create = st.button("Create Company", use_container_width=True)

                if btn_create:
                    if not new_company_name.strip():
                        st.warning("Please enter a company name.")
                    else:
                        clean_name = sanitize_company_name(new_company_name)
                        get_company_pdf_dir(clean_name)
                        get_company_index_dir(clean_name)
                        st.session_state.selected_company = clean_name
                        clear_app_caches()
                        st.success(f"Created company workspace '{clean_name}'.")
                        st.rerun()

            with tab_files:
                pdf_dir = get_company_pdf_dir(active_company)
                current_pdfs = list_policy_pdfs(pdf_dir)
                if not current_pdfs:
                    st.caption(f"No PDFs found for {active_company}.")
                else:
                    for pdf_path in current_pdfs:
                        size_kb = pdf_path.stat().st_size / 1024
                        st.markdown(
                            f"""
                            <div class="doc-item">
                                <span style="color:#ffffff; font-family:ui-monospace, monospace; font-weight:500;">{html.escape(pdf_path.name)}</span>
                                <span style="color:#8e8e98;">{size_kb:.1f} KB</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
                col_a, col_b = st.columns(2)
                with col_a:
                    if active_company == DEFAULT_COMPANY:
                        if st.button("Restore Default Demo PDFs", use_container_width=True):
                            with st.spinner("Restoring demo PDFs..."):
                                pdf_dir = get_company_pdf_dir(DEFAULT_COMPANY)
                                index_dir = get_company_index_dir(DEFAULT_COMPANY)
                                for old in pdf_dir.glob("*.pdf"):
                                    old.unlink()
                                for policy in MOCK_POLICIES:
                                    build_policy_pdf(policy, pdf_dir)
                                build_faiss_index(pdf_dir=pdf_dir, index_dir=index_dir, force=True)
                                clear_app_caches()
                            st.success("Default demo policies restored.")
                            st.rerun()
                with col_b:
                    if active_company != DEFAULT_COMPANY:
                        if st.button(f"Delete Company '{active_company}'", use_container_width=True):
                            shutil.rmtree(get_company_pdf_dir(active_company), ignore_errors=True)
                            shutil.rmtree(get_company_index_dir(active_company), ignore_errors=True)
                            st.session_state.selected_company = DEFAULT_COMPANY
                            clear_app_caches()
                            st.warning(f"Deleted workspace for {active_company}.")
                            st.rerun()


def run_answer(
    question: str,
    mode: str,
    top_k: int,
    threshold: float,
    selected_model: Optional[str],
    company: str,
) -> AnswerResult:
    index_dir = get_company_index_dir(company)
    if mode == STRICT_MODE:
        return answer_strict_retrieval(
            question=question,
            top_k=top_k,
            min_relevance_score=threshold,
            index_dir=index_dir,
        )
    return answer_rag_synthesis(
        question=question,
        top_k=top_k,
        min_relevance_score=threshold,
        model=selected_model,
        index_dir=index_dir,
        company_name=company,
    )


def render_answer_card(
    answer_text: str,
    company: str,
    view_mode: str,
    mode: str,
    citations: List[str],
    model: Optional[str] = None,
    refused: bool = False,
) -> None:
    st.markdown('<div class="section-label">Answer</div>', unsafe_allow_html=True)

    if refused:
        badge_text = "Insufficient Evidence"
    elif mode == STRICT_MODE:
        badge_text = f"Verified Excerpts &mdash; {html.escape(company)}" if view_mode == USER_MODE else f"Strict Retrieval &mdash; {html.escape(company)}"
    else:
        badge_text = f"Policy Answer &mdash; {html.escape(company)}" if view_mode == USER_MODE else f"RAG Synthesis &mdash; {html.escape(company)} ({html.escape(model or 'Ollama')})"

    escaped_body = html.escape(answer_text).replace("\n", "<br>")

    st.markdown(
        f"""
        <div class="answer-card">
            <div class="badge-pill">{badge_text}</div>
            <div class="answer-body">{escaped_body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">Sources</div>', unsafe_allow_html=True)
    if citations:
        pill_items = "".join(
            f'<div class="citation-chip">{html.escape(c)}</div>'
            for c in citations
        )
        st.markdown(f'<div class="citation-wrap">{pill_items}</div>', unsafe_allow_html=True)
    else:
        st.caption("No sources met the threshold.")


def render_retrieved_chunks(result: AnswerResult) -> None:
    st.markdown('<div class="section-label">Developer Telemetry: Retrieved Chunks</div>', unsafe_allow_html=True)
    if not result.retrieved_chunks:
        st.caption("No chunks were retrieved.")
        return

    for idx, chunk in enumerate(result.retrieved_chunks, start=1):
        score_pct = int(chunk.relevance_score * 100)
        title = f"{idx}. {chunk.citation} &mdash; Score: {chunk.relevance_score:.2f} ({score_pct}%)"
        with st.expander(title, expanded=(idx == 1)):
            document_name = html.escape(str(chunk.metadata.get("document_name", "Unknown")))
            st.markdown(
                f"""
                <div style="font-size:0.8rem; color:#c5c5d0; margin-bottom:0.75rem;">
                    Document: <strong style="color:#ffffff;">{document_name}</strong> &bull; Distance: {chunk.distance:.3f} &bull; Page: {chunk.metadata.get('page_number', '?')}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.12); border-radius:8px; padding:0.9rem; font-size:0.95rem; line-height:1.6; color:#ffffff;">
                    {html.escape(truncate_text(chunk.text, max_chars=1600))}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
            st.json(chunk.metadata, expanded=False)


def main() -> None:
    st.set_page_config(page_title="Policy Intelligence", layout="wide")
    render_styles()
    initialize_session_state()

    controls = render_sidebar()
    active_company = str(controls["company"])
    view_mode = str(controls["view_mode"])

    render_hero(active_company, view_mode)
    render_status_panel(controls)

    render_document_management(active_company, view_mode)

    st.markdown('<div class="section-label">Question</div>', unsafe_allow_html=True)
    question = st.text_area(
        "Question",
        key="question",
        height=95,
        label_visibility="collapsed",
        placeholder=f"Ask a question about {active_company}'s policies...",
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        ask_clicked = st.button("Search Policies", type="primary", use_container_width=True)

    if ask_clicked:
        if not question.strip():
            st.warning("Please enter a question first.")
            return

        try:
            with st.spinner("Searching neural index..."):
                result = run_answer(
                    question=question,
                    mode=str(controls["mode"]),
                    top_k=int(controls["top_k"]),
                    threshold=float(controls["threshold"]),
                    selected_model=controls["selected_model"],
                    company=active_company,
                )
        except FileNotFoundError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"Error: {exc}")
            return

        # If RAG synthesis was performed and we have a prompt, stream tokens live!
        if result.mode == RAG_MODE and not result.refused and result.prompt and result.model:
            st.markdown('<div class="section-label">Answer</div>', unsafe_allow_html=True)
            badge_text = f"Policy Answer &mdash; {html.escape(active_company)}" if view_mode == USER_MODE else f"RAG Synthesis &mdash; {html.escape(active_company)} ({html.escape(result.model)})"

            # Container for streamed tokens
            answer_box = st.empty()
            accumulated = []

            try:
                for token in stream_with_ollama(prompt=result.prompt, model=result.model):
                    accumulated.append(token)
                    text_so_far = "".join(accumulated)
                    escaped = html.escape(text_so_far).replace("\n", "<br>")
                    answer_box.markdown(
                        f"""
                        <div class="answer-card">
                            <div class="badge-pill">{badge_text}</div>
                            <div class="answer-body">{escaped}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                final_answer = "".join(accumulated) or result.answer
            except Exception:
                final_answer = result.answer
                escaped = html.escape(final_answer).replace("\n", "<br>")
                answer_box.markdown(
                    f"""
                    <div class="answer-card">
                        <div class="badge-pill">{badge_text}</div>
                        <div class="answer-body">{escaped}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Sources
            st.markdown('<div class="section-label">Sources</div>', unsafe_allow_html=True)
            if result.citations:
                pill_items = "".join(
                    f'<div class="citation-chip">{html.escape(c)}</div>'
                    for c in result.citations
                )
                st.markdown(f'<div class="citation-wrap">{pill_items}</div>', unsafe_allow_html=True)
            else:
                st.caption("No sources met the threshold.")

        else:
            # Fallback or Strict Mode (instant quote)
            render_answer_card(
                answer_text=result.answer,
                company=active_company,
                view_mode=view_mode,
                mode=result.mode,
                citations=result.citations,
                model=result.model,
                refused=result.refused,
            )

        # Developer Mode telemetry
        if view_mode == DEV_MODE:
            render_retrieved_chunks(result)

    else:
        hint_text = "Select an example question from the sidebar or enter a prompt." if view_mode == USER_MODE else "Adjust sliders in the sidebar, inspect chunks, or enter a query."
        st.markdown(
            f"""
            <div style="font-size:0.84rem; color:#8e8e98; margin-top:0.4rem;">
                {hint_text}
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
