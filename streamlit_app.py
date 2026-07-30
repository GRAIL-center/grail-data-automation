from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import streamlit as st
import yaml

from src.collect_comments.fetch_comments import processComments
from src.collect_notices.collect import collectNotices
from src.services.ai_client import reset_client
from src.services.config import CONFIG_PATH, loadConfig, loadNoticeConfig

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"


@dataclass
class RunState:
    identifier: str
    label: str
    logger_name: str
    started_at: float = field(default_factory=time.time)
    log_queue: queue.Queue[str | None] = field(default_factory=queue.Queue)
    logs: list[str] = field(default_factory=list)
    completed: bool = False
    succeeded: bool = False
    message: str = "Starting..."


class QueueLogHandler(logging.Handler):
    def __init__(self, log_queue: queue.Queue[str | None]) -> None:
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        self.log_queue.put(self.format(record))


def split_terms(value: str) -> list[str]:
    return [term.strip() for term in value.splitlines() if term.strip()]


def start_run(
    label: str,
    logger_name: str,
    action: Callable[[], Any],
    success_message: Callable[[Any], str],
) -> bool:
    active_run = st.session_state.get("active_run")
    if active_run is not None and not getattr(active_run, "completed", True):
        st.warning("Wait for the active collection run to finish before starting another.")
        return False

    run = RunState(
        identifier=uuid.uuid4().hex,
        label=label,
        logger_name=logger_name,
    )
    run.logs.append("INFO: Run queued. Waiting for the collection worker to start.")
    st.session_state.active_run = run

    def worker() -> None:
        logger = logging.getLogger(logger_name)
        previous_level = logger.level
        handler = QueueLogHandler(run.log_queue)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        try:
            logger.info("Starting %s", label.lower())
            result = action()
            run.succeeded = True
            run.message = success_message(result)
            logger.info(run.message)
        except Exception:
            run.succeeded = False
            run.message = f"{label} failed. Review the log for details."
            logger.exception("%s failed", label)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)
            handler.close()
            run.completed = True
            run.log_queue.put(None)

    threading.Thread(target=worker, daemon=True, name=f"streamlit-{label}").start()
    return True


def drain_active_run() -> RunState | None:
    run = st.session_state.get("active_run")
    required_attributes = ("log_queue", "logs", "label", "completed", "succeeded", "message")
    if run is None or not all(hasattr(run, attribute) for attribute in required_attributes):
        return None

    while True:
        try:
            message = run.log_queue.get_nowait()
        except queue.Empty:
            break

        if message is not None:
            run.logs.append(message)

    return run


@st.fragment(run_every=1)
def show_run_monitor(run_label: str) -> None:
    run = drain_active_run()
    matching_run = run if run is not None and run.label == run_label else None
    monitor_title = f"{run_label} monitor"

    with st.container(border=True):
        title_col, status_col = st.columns([4, 1])
        with title_col:
            st.subheader(monitor_title)
        with status_col:
            if matching_run is None:
                st.metric("Status", "Ready")
            elif matching_run.completed and matching_run.succeeded:
                st.metric("Status", "Complete")
            elif matching_run.completed:
                st.metric("Status", "Failed")
            else:
                elapsed_seconds = int(time.time() - matching_run.started_at)
                st.metric("Running", f"{elapsed_seconds}s")

        if matching_run is None:
            st.caption("Start a run below. Live logs will appear here immediately.")
            st.code("No run has started yet.", language="text")
            return

        log_text = "\n".join(matching_run.logs) or "Waiting for pipeline output..."
        if matching_run.completed and matching_run.succeeded:
            st.success(matching_run.message)
        elif matching_run.completed:
            st.error(matching_run.message)
        else:
            st.info("The pipeline is running. This monitor refreshes every second.")

        st.code(log_text, language="text")

        if matching_run.completed:
            download_col, clear_col, _ = st.columns([1, 1, 4])
            with download_col:
                st.download_button(
                    "Download log",
                    data=log_text,
                    file_name=f"{matching_run.identifier}.log",
                    mime="text/plain",
                )
            with clear_col:
                if st.button("Clear log", key=f"clear-run-{matching_run.identifier}"):
                    st.session_state.active_run = None
                    st.rerun()


def show_dashboard() -> None:
    st.title("GRAIL workspace")
    st.caption("Federal Register notice discovery and public-comment processing")

    try:
        config = loadConfig() or {}
    except (OSError, yaml.YAMLError) as error:
        st.error(f"Unable to load config.yaml: {error}")
        return

    notice_settings = config.get("notice_collection_settings", {})
    notice_terms = config.get("notice_search_terms", {})
    default_terms = notice_terms.get("default_terms", []) if isinstance(notice_terms, dict) else []
    search_terms = notice_terms.get("search_terms", []) if isinstance(notice_terms, dict) else []

    notices_col, terms_col, artifacts_col = st.columns(3)
    notices_col.metric("Notice limit", notice_settings.get("max_notices", "—"))
    terms_col.metric("Configured search terms", len(default_terms) + len(search_terms))
    artifacts_col.metric(
        "Downloaded FR folders",
        len([path for path in DATA_DIR.iterdir() if path.is_dir()]) if DATA_DIR.exists() else 0,
    )

    st.subheader("How this interface works")
    st.markdown(
        """
        - **Notices** runs the existing Federal Register collection workflow.
        - **Comments** runs the existing comment workflow and stores artifacts under
          `data/<FR number>/<comment ID>/`.
        - **Settings** validates and saves `config.yaml`; `.env` remains the place
          for API keys and Google Sheet URLs.
        """
    )


def show_notices() -> None:
    st.title("Notices")
    st.caption("Configure and run the existing notice collection workflow.")
    show_run_monitor("Notice collection")

    try:
        settings, terms = loadNoticeConfig()
    except (OSError, yaml.YAMLError) as error:
        st.error(f"Unable to load notice configuration: {error}")
        return

    settings = settings or {}
    terms = terms or {}
    order_options = {
        "Relevance": "relevance",
        "Newest first": "newest",
        "Oldest first": "oldest",
        "Executive order number": "executive_order_number",
    }
    selected_order = settings.get("order", "relevance")
    selected_order_label = next(
        (label for label, value in order_options.items() if value == selected_order),
        "Relevance",
    )

    try:
        configured_start_date = date.fromisoformat(
            str(settings.get("start_date", "2021-01-01"))
        )
    except ValueError:
        configured_start_date = date(2021, 1, 1)

    with st.form("notice-collection-form", border=False):
        settings_col, terms_col = st.columns(2, gap="large")

        with settings_col:
            st.subheader("Collection settings")
            max_notices = st.number_input(
                "Max notices",
                min_value=1,
                value=int(settings.get("max_notices", 5)),
                step=1,
            )
            docket_type = st.text_input(
                "Docket type",
                value=str(settings.get("docket_type", "NOTICE")),
            )
            order_label = st.selectbox(
                "Order",
                options=list(order_options),
                index=list(order_options).index(selected_order_label),
            )
            start_date = st.date_input("Start date", value=configured_start_date)

        with terms_col:
            st.subheader("Search terms")
            st.caption("Enter one term per line. Both lists are used in the run.")
            default_terms = st.text_area(
                "Default terms",
                value="\n".join(terms.get("default_terms", []) or []),
                height=220,
            )
            search_terms = st.text_area(
                "Search terms",
                value="\n".join(terms.get("search_terms", []) or []),
                height=180,
            )

        submitted = st.form_submit_button("Run notice collection", type="primary")

    if submitted:
        run_settings = {
            "max_notices": int(max_notices),
            "docket_type": docket_type.strip(),
            "order": order_options[order_label],
            "start_date": start_date.isoformat(),
        }
        run_terms = {
            "default_terms": split_terms(default_terms),
            "search_terms": split_terms(search_terms),
        }
        if start_run(
            "Notice collection",
            "src.collect_notices.collect",
            lambda: collectNotices(run_settings, run_terms),
            lambda _: "Notice collection finished.",
        ):
            st.rerun()


def show_comments() -> None:
    st.title("Public comments")
    st.caption("Process comments for a Federal Register document or Regulations.gov docket.")
    show_run_monitor("Comment collection")

    with st.form("comment-collection-form", border=False):
        fr_number = st.text_input(
            "Federal Register number or docket ID",
            placeholder="Example: 2024-12345 or NIST-2021-0003",
        )
        spreadsheet_url = st.text_input(
            "Spreadsheet URL (optional)",
            placeholder="Uses COMMENT_SHEET_URL when left blank",
        )
        submitted = st.form_submit_button("Process comments", type="primary")

    if submitted:
        identifier = fr_number.strip()
        if not identifier:
            st.error("Enter a Federal Register number or Regulations.gov docket ID.")
            return

        override_url = spreadsheet_url.strip() or None
        if start_run(
            "Comment collection",
            "src.collect_comments",
            lambda: processComments(identifier, override_url),
            lambda count: f"Processed {count} comment(s).",
        ):
            st.rerun()

    st.subheader("Downloaded artifacts")
    if not DATA_DIR.exists():
        st.info("No comment artifacts have been downloaded yet.")
        return

    fr_folders = sorted(path for path in DATA_DIR.iterdir() if path.is_dir())
    if not fr_folders:
        st.info("No comment artifacts have been downloaded yet.")
        return

    selected_fr = st.selectbox("FR number", options=fr_folders, format_func=lambda path: path.name)
    comment_folders = sorted(path for path in selected_fr.iterdir() if path.is_dir())
    if not comment_folders:
        st.info("No downloaded comments are available in this folder.")
        return

    selected_comment = st.selectbox(
        "Comment ID",
        options=comment_folders,
        format_func=lambda path: path.name,
    )
    manifest_path = selected_comment / "manifest.json"
    full_text_path = selected_comment / "full_comment.txt"

    if manifest_path.is_file():
        try:
            st.json(json.loads(manifest_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as error:
            st.warning(f"Unable to read the artifact manifest: {error}")
    if full_text_path.is_file():
        try:
            full_text = full_text_path.read_text(encoding="utf-8")
        except OSError as error:
            st.warning(f"Unable to read the full comment text: {error}")
        else:
            st.text_area("Full comment text", value=full_text, height=320, disabled=True)
            st.download_button(
                "Download full comment text",
                data=full_text,
                file_name=f"{selected_fr.name}-{selected_comment.name}.txt",
                mime="text/plain",
            )


def save_config(config_text: str) -> tuple[bool, str]:
    try:
        config_data = yaml.safe_load(config_text)
    except yaml.YAMLError as error:
        return False, f"Invalid YAML: {error}"

    if not isinstance(config_data, dict):
        return False, "The top-level YAML value must be a mapping."

    config_path = Path(CONFIG_PATH)
    temporary_path = config_path.with_name(f"{config_path.name}.tmp")

    try:
        temporary_path.write_text(config_text, encoding="utf-8")
        temporary_path.replace(config_path)
    except OSError as error:
        temporary_path.unlink(missing_ok=True)
        return False, f"Unable to save config.yaml: {error}"

    reset_client()
    return True, "config.yaml saved. New AI settings will apply to future runs."


def show_settings() -> None:
    st.title("Settings")
    st.caption("Edit configuration safely. API keys and sheet URLs remain in .env.")

    config_path = Path(CONFIG_PATH)
    try:
        config_text = config_path.read_text(encoding="utf-8")
    except OSError as error:
        st.error(f"Unable to read config.yaml: {error}")
        return

    with st.form("settings-form", border=False):
        edited_config = st.text_area(
            "config.yaml",
            value=config_text,
            height=620,
            help="The file must contain valid YAML with a top-level mapping.",
        )
        submitted = st.form_submit_button("Save configuration", type="primary")

    if submitted:
        saved, message = save_config(edited_config)
        if saved:
            st.success(message)
        else:
            st.error(message)


def main() -> None:
    st.set_page_config(
        page_title="GRAIL",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
            :root {
                color-scheme: light;
                --grail-ink: #1c1c1c;
                --grail-muted: #5f6368;
                --grail-line: #dedede;
                --grail-surface: #ffffff;
                --grail-canvas: #f7f7f5;
                --grail-accent: #1f4d3d;
                --grail-accent-hover: #163a2e;
                --grail-code: #202321;
            }

            html, body {
                background: var(--grail-canvas);
                color: var(--grail-ink);
            }

            [data-testid="stAppViewContainer"] {
                background: var(--grail-canvas);
                color: var(--grail-ink);
            }

            [data-testid="stHeader"] {
                background: transparent;
            }

            [data-testid="stMainBlockContainer"] {
                max-width: 1180px;
                padding-top: 3.25rem;
                padding-bottom: 3.5rem;
            }

            h1, h2, h3, p, label, [data-testid="stMarkdownContainer"],
            [data-testid="stCaptionContainer"], [data-testid="stMetricLabel"],
            [data-testid="stMetricValue"] {
                color: var(--grail-ink);
            }

            [data-testid="stCaptionContainer"], .stCaption, small {
                color: var(--grail-muted) !important;
            }

            h1 {
                font-size: 2.15rem !important;
                font-weight: 700 !important;
                letter-spacing: -0.045em;
                margin-bottom: 0.35rem !important;
            }

            h2, h3 {
                letter-spacing: -0.025em;
            }

            [data-testid="stSidebar"] {
                background: var(--grail-surface);
                border-right: 1px solid var(--grail-line);
            }

            [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
                padding-top: 1.5rem;
            }

            [data-testid="stSidebar"] h1 {
                letter-spacing: -0.05em;
            }

            [data-testid="stSidebar"] [role="radiogroup"] label {
                border-radius: 7px;
                color: var(--grail-ink) !important;
                padding: 0.35rem 0.45rem;
            }

            [data-testid="stSidebar"] [role="radiogroup"] label:hover {
                background: #f0f1ee;
            }

            [data-testid="stForm"] {
                border: 1px solid var(--grail-line);
                border-radius: 12px;
                background: var(--grail-surface);
                padding: 1.5rem;
            }

            [data-baseweb="input"] > div,
            [data-baseweb="select"] > div,
            [data-baseweb="textarea"] textarea {
                border-color: #c9cbc7 !important;
                border-radius: 7px !important;
                background: #ffffff !important;
                color: var(--grail-ink) !important;
            }

            [data-baseweb="input"] input,
            [data-baseweb="textarea"] textarea,
            [data-baseweb="select"] input {
                color: var(--grail-ink) !important;
                -webkit-text-fill-color: var(--grail-ink) !important;
            }

            [data-baseweb="input"] > div:focus-within,
            [data-baseweb="select"] > div:focus-within,
            [data-baseweb="textarea"] textarea:focus {
                border-color: var(--grail-accent) !important;
                box-shadow: 0 0 0 3px rgba(31, 77, 61, 0.14) !important;
            }

            [data-baseweb="textarea"] textarea {
                font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace !important;
                line-height: 1.5;
            }

            .stButton > button, .stFormSubmitButton > button {
                min-height: 2.5rem;
                border: 0 !important;
                border-radius: 7px !important;
                background: var(--grail-accent) !important;
                color: #ffffff !important;
                font-weight: 650;
                padding: 0.45rem 1rem;
                transition: background 120ms ease, transform 120ms ease;
            }

            .stButton > button:hover, .stFormSubmitButton > button:hover {
                background: var(--grail-accent-hover) !important;
                color: #ffffff !important;
                transform: translateY(-1px);
            }

            .stButton > button:disabled, .stFormSubmitButton > button:disabled {
                background: #a8aaa6 !important;
                color: #ffffff !important;
            }

            [data-testid="stMetric"] {
                border: 1px solid var(--grail-line);
                border-radius: 10px;
                padding: 1.1rem 1.2rem;
                background: var(--grail-surface);
                box-shadow: 0 1px 2px rgba(19, 32, 26, 0.04);
            }

            [data-testid="stVerticalBlockBorderWrapper"] {
                border: 1px solid var(--grail-line);
                border-left: 3px solid var(--grail-accent);
                border-radius: 10px;
                background: var(--grail-surface);
                box-shadow: 0 6px 18px rgba(19, 32, 26, 0.05);
            }

            [data-testid="stMetricLabel"] {
                color: var(--grail-muted) !important;
                font-size: 0.82rem;
            }

            [data-testid="stMetricValue"] {
                color: var(--grail-ink) !important;
                font-size: 1.55rem;
            }

            [data-testid="stStatusWidget"] {
                border: 1px solid var(--grail-line);
                border-radius: 10px;
                background: var(--grail-surface);
            }

            [data-testid="stCodeBlock"] pre,
            [data-testid="stCode"] pre {
                border-radius: 7px;
                background: var(--grail-code) !important;
                color: #eff3ed !important;
                border: 1px solid #363b37;
            }

            [data-testid="stCodeBlock"] pre *,
            [data-testid="stCode"] pre * {
                color: #eff3ed !important;
            }

            [data-testid="stAlert"] {
                border-radius: 8px;
            }

            @media (max-width: 760px) {
                [data-testid="stMainBlockContainer"] {
                    padding-top: 2rem;
                    padding-left: 1rem;
                    padding-right: 1rem;
                }

                h1 {
                    font-size: 1.85rem !important;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "active_run" not in st.session_state:
        st.session_state.active_run = None

    with st.sidebar:
        st.title("GRAIL")
        st.caption("Collection workspace")
        page = st.radio("Navigate", ["Dashboard", "Notices", "Comments", "Settings"])
        st.divider()
        st.caption("Streamlit preview interface")

    if page == "Dashboard":
        show_dashboard()
    elif page == "Notices":
        show_notices()
    elif page == "Comments":
        show_comments()
    else:
        show_settings()


if __name__ == "__main__":
    main()
