from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

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
) -> None:
    run = RunState(
        identifier=uuid.uuid4().hex,
        label=label,
        logger_name=logger_name,
    )
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


def drain_active_run() -> RunState | None:
    run = st.session_state.get("active_run")
    if not isinstance(run, RunState):
        return None

    while True:
        try:
            message = run.log_queue.get_nowait()
        except queue.Empty:
            break

        if message is not None:
            run.logs.append(message)

    return run


def show_active_run() -> None:
    run = drain_active_run()
    if run is None:
        return

    status = "complete" if run.completed else "running"
    state = "complete" if run.completed and run.succeeded else "error" if run.completed else "running"

    with st.status(run.label, state=state, expanded=True) as status_panel:
        status_panel.write(run.message)
        st.code("\n".join(run.logs) or "Waiting for pipeline output...", language="text")
        if run.completed:
            status_panel.update(label=run.message, state=state, expanded=True)
        else:
            status_panel.update(label=f"{run.label} is running", state=status, expanded=True)

    if not run.completed:
        time.sleep(1)
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
            start_date = st.date_input(
                "Start date",
                value=str(settings.get("start_date", "2021-01-01")),
            )

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
        start_run(
            "Notice collection",
            "src.collect_notices.collect",
            lambda: collectNotices(run_settings, run_terms),
            lambda _: "Notice collection finished.",
        )
        st.rerun()


def show_comments() -> None:
    st.title("Public comments")
    st.caption("Process comments for a Federal Register document or Regulations.gov docket.")

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
        start_run(
            "Comment collection",
            "src.collect_comments",
            lambda: processComments(identifier, override_url),
            lambda count: f"Processed {count} comment(s).",
        )
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
        st.json(json.loads(manifest_path.read_text(encoding="utf-8")))
    if full_text_path.is_file():
        full_text = full_text_path.read_text(encoding="utf-8")
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
            .stApp { background: #fafafa; }
            [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e5e5e5; }
            [data-testid="stSidebar"] h1 { letter-spacing: -0.04em; }
            .stButton > button, .stFormSubmitButton > button {
                border: 0; border-radius: 6px; background: #171717; color: #ffffff;
                font-weight: 600;
            }
            .stButton > button:hover, .stFormSubmitButton > button:hover {
                background: #404040; color: #ffffff; border: 0;
            }
            [data-testid="stMetric"] { border: 1px solid #e5e5e5; border-radius: 10px; padding: 16px; background: #ffffff; }
            textarea { font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace !important; }
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

    show_active_run()


if __name__ == "__main__":
    main()
