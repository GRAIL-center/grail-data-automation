"""Local Flask control panel for the connected GRAIL workflows."""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from src.audit import create_run_id, utc_now, validate_run_id
from src.collect_notices.pipeline import runNoticePipeline
from src.config import (
    loadConfig,
    resolveConfigPath,
    saveConfigText,
    validateFederalRegisterNumber,
)
from src.logging_config import configure_logging, log_context
from src.pipeline import collectAndAnalyzeComments, runConnectedWorkflow

logger = logging.getLogger(__name__)
FAILURE_PREVIEW_LIMIT = 25


def _utc_datetime() -> datetime:
    return datetime.now(timezone.utc)


def _compact_result(workflow: str, result: dict[str, Any]) -> dict[str, Any]:
    failures = result.get("failures", [])
    common = {
        "run_id": result.get("run_id"),
        "manifest_path": (
            result.get("run_manifest_path")
            or result.get("manifest_path")
        ),
        "audit_path": result.get("audit_path"),
        "failure_count": len(failures),
        "failures": failures[:FAILURE_PREVIEW_LIMIT],
        "failures_truncated": max(
            0,
            len(failures) - FAILURE_PREVIEW_LIMIT,
        ),
    }
    if workflow == "notices":
        return {
            **common,
            "status": result.get("status"),
            "candidate_count": result.get("candidate_count", 0),
            "selected_count": result.get("selected_count", 0),
            "processed_count": len(result.get("records", [])),
            "review_required_count": sum(
                1
                for record in result.get("records", [])
                if record.get("analysis", {}).get("review", {}).get("required")
            ),
            "sheet_rows_appended": result.get("sheet_rows_appended", 0),
            "sheet_rows_skipped": result.get("sheet_rows_skipped", 0),
            "sheet_worksheet_title": result.get("sheet_worksheet_title"),
            "sheet_export": result.get("sheet_export"),
            "notices": [
                {
                    "document_number": record.get("document_number"),
                    "title": record.get("metadata", {}).get("title"),
                    "review_required": record.get("analysis", {})
                    .get("review", {})
                    .get("required"),
                    "analysis_path": record.get("analysis_path"),
                }
                for record in result.get("records", [])
            ],
        }
    if workflow == "comments":
        return {
            **common,
            "status": result.get("status"),
            "fr_document_number": result.get("fr_document_number"),
            "notice_title": result.get("notice_title"),
            "collected_comment_count": result.get("collected_comment_count", 0),
            "analyzed_comment_count": len(result.get("records", [])),
            "degraded_comment_count": result.get(
                "degraded_comment_count",
                0,
            ),
            "failed_comment_count": result.get(
                "failed_comment_count",
                0,
            ),
            "review_required_count": sum(
                1
                for record in result.get("records", [])
                if record.get("analysis_record", {})
                .get("analysis", {})
                .get("review", {})
                .get("required")
            ),
            "sheet_rows_appended": result.get("sheet_rows_appended", 0),
            "sheet_rows_updated": result.get("sheet_rows_updated", 0),
            "sheet_rows_skipped": result.get("sheet_rows_skipped", 0),
            "sheet_worksheet_title": result.get("sheet_worksheet_title"),
            "sheet_export": result.get("sheet_export"),
        }
    return {
        **common,
        "notice_run": result.get("notice_run"),
        "selected_document_numbers": result.get("selected_document_numbers", []),
        "comment_runs": result.get("comment_runs", []),
        "status": result.get("status"),
    }


class JobManager:
    """Small in-memory background job registry backed by application artifacts."""

    def __init__(self, max_workers: int = 2, history_limit: int = 100) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="grail-web",
        )
        self._history_limit = history_limit
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def submit(
        self,
        workflow: str,
        run_id: str,
        function: Callable[..., dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        job = {
            "job_id": job_id,
            "workflow": workflow,
            "run_id": run_id,
            "status": "queued",
            "created_at": utc_now(),
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = job
            self._trim()

        def execute() -> None:
            with self._lock:
                job["status"] = "running"
                job["started_at"] = utc_now()
            try:
                with log_context(
                    run_id=run_id,
                    workflow=workflow,
                    source="web",
                ):
                    result = function(**kwargs)
                compact = _compact_result(workflow, result)
            except Exception as exc:
                logger.exception("Web job %s failed", job_id)
                with self._lock:
                    job["status"] = "failed"
                    job["error"] = {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    job["finished_at"] = utc_now()
                return
            with self._lock:
                job["status"] = (
                    "completed_with_failures"
                    if compact.get("failure_count")
                    else "completed"
                )
                job["result"] = compact
                job["finished_at"] = utc_now()

        self._executor.submit(execute)
        return self.get(job_id) or job

    def _trim(self) -> None:
        if len(self._jobs) <= self._history_limit:
            return
        completed = [
            key
            for key, value in self._jobs.items()
            if value["status"] not in {"queued", "running"}
        ]
        for key in completed[: max(0, len(self._jobs) - self._history_limit)]:
            self._jobs.pop(key, None)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._jobs.get(job_id)
            return json.loads(json.dumps(value, default=str)) if value else None

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            values = json.loads(json.dumps(list(self._jobs.values()), default=str))
        return list(reversed(values))

    def shutdown(self, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=False)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _recent_runs(root: Path, limit: int) -> list[dict[str, Any]]:
    runs_dir = root / "runs"
    if not runs_dir.is_dir():
        return []
    candidates = []
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        path = run_dir / "run_manifest.json"
        connected = run_dir / "connected_workflow.json"
        selected = path if path.is_file() else connected
        if not selected.is_file():
            continue
        payload = _load_json(selected)
        if payload is None:
            continue
        candidates.append(
            {
                "run_id": payload.get("run_id") or run_dir.name,
                "workflow": payload.get("workflow") or (
                    "comments"
                    if payload.get("pipeline")
                    or payload.get("input", {}).get("fr_document_number")
                    else "unknown"
                ),
                "status": payload.get("status", "unknown"),
                "started_at": payload.get("started_at"),
                "finished_at": payload.get("finished_at"),
                "manifest_path": str(selected),
                "mtime": selected.stat().st_mtime,
            }
        )
    candidates.sort(key=lambda value: value["mtime"], reverse=True)
    return candidates[:limit]


def _latest_notices(root: Path) -> list[dict[str, Any]]:
    payload = _load_json(root / "notice_batch.json")
    if not payload:
        return []
    rows = payload.get("rows")
    return rows if isinstance(rows, list) else []


def _optional_int(name: str, minimum: int = 1) -> int | None:
    value = request.form.get(name, "").strip()
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name.replace('_', ' ')} must be an integer") from exc
    if parsed < minimum:
        raise ValueError(
            f"{name.replace('_', ' ')} must be at least {minimum}"
        )
    return parsed


def _optional_float(name: str, minimum: float, maximum: float) -> float | None:
    value = request.form.get(name, "").strip()
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name.replace('_', ' ')} must be a number") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(
            f"{name.replace('_', ' ')} must be between {minimum} and {maximum}"
        )
    return parsed


def createApp(
    configPath: str = "config.yaml",
    *,
    jobManager: JobManager | None = None,
) -> Flask:
    config_path = Path(configPath).resolve()
    config = loadConfig(config_path)
    web_settings = config["web"]

    def resolved_paths() -> tuple[Path, Path]:
        current = loadConfig(config_path)
        application = current["application"]
        current_artifact_root = resolveConfigPath(
            config_path,
            application["artifact_root"],
        )
        current_log_dir = resolveConfigPath(
            config_path,
            application["log_dir"],
        )
        return current_artifact_root, current_log_dir

    artifact_root, log_dir = resolved_paths()

    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=(
            os.getenv(web_settings["secret_key_env"])
            or secrets.token_hex(32)
        ),
        MAX_CONTENT_LENGTH=1_000_000,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        GRAIL_CONFIG_PATH=str(config_path),
        GRAIL_ARTIFACT_ROOT=str(artifact_root),
        GRAIL_LOG_DIR=str(log_dir),
    )
    jobs = jobManager or JobManager(web_settings["background_workers"])
    app.extensions["grail_jobs"] = jobs

    @app.before_request
    def csrf_protection() -> None:
        token = session.setdefault("_csrf_token", secrets.token_urlsafe(32))
        if request.method == "POST":
            submitted = request.form.get("_csrf_token", "")
            if not submitted or not secrets.compare_digest(token, submitted):
                abort(400, "Invalid form token; refresh the page and try again.")

    @app.context_processor
    def shared_template_values() -> dict[str, Any]:
        return {
            "csrf_token": session.get("_csrf_token", ""),
            "current_year": _utc_datetime().year,
        }

    @app.get("/")
    def dashboard():
        current = loadConfig(config_path)
        current_artifact_root, _ = resolved_paths()
        latest = _latest_notices(current_artifact_root)
        primary_ai_ready = (
            current["ai"]["provider"] == "ollama"
            or bool(
                os.getenv(current["ai"]["openrouter"]["api_key_env"])
            )
        )
        fallback = current["ai"].get("fallback")
        fallback_ai_ready = bool(
            isinstance(fallback, dict)
            and (
                fallback.get("provider") == "ollama"
                or (
                    fallback.get("provider") == "openrouter"
                    and os.getenv(
                        current["ai"]["openrouter"]["api_key_env"]
                    )
                )
            )
        )
        environment = {
            "Regulations.gov key": bool(os.getenv("REGULATION_API_KEY")),
            "AI provider": primary_ai_ready or fallback_ai_ready,
            "Google service account": (
                config_path.parent / current["application"]["service_account_file"]
            ).is_file(),
            "Notice sheet URL": bool(
                os.getenv(current["notices"]["output"]["sheet_url_env"])
                or os.getenv(current["application"]["sheet_url_env"])
            ),
            "Comment sheet URL": bool(
                os.getenv(current["comments"]["output"]["sheet_url_env"])
                or os.getenv(current["application"]["sheet_url_env"])
            ),
        }
        return render_template(
            "dashboard.html",
            config=current,
            jobs=jobs.list()[:10],
            recent_runs=_recent_runs(
                current_artifact_root,
                current["web"]["recent_run_limit"],
            ),
            notices=latest,
            environment=environment,
        )

    @app.post("/run/notices")
    def run_notices():
        try:
            current_artifact_root, _ = resolved_paths()
            max_notices = _optional_int("max_notices")
            workers = _optional_int("workers")
            raw_queries = request.form.get("query_terms", "")
            queries = [
                line.strip()
                for line in raw_queries.splitlines()
                if line.strip()
            ] or None
            run_id = create_run_id()
            job = jobs.submit(
                "notices",
                run_id,
                runNoticePipeline,
                configPath=str(config_path),
                downloadRoot=str(current_artifact_root),
                spreadsheetUrl=request.form.get("sheet_url", "").strip() or None,
                useConfiguredSheet="configured_sheet" in request.form,
                queryTerms=queries,
                maxResults=max_notices,
                maxWorkers=workers,
                skipAI="skip_ai" in request.form,
                comparisonModels=(
                    []
                    if "no_comparison" in request.form
                    else None
                ),
                runId=run_id,
            )
        except ValueError as exc:
            flash(str(exc), "error")
            return redirect(url_for("dashboard"))
        flash("Notice discovery was queued.", "success")
        return redirect(url_for("job_detail", job_id=job["job_id"]))

    @app.post("/run/comments")
    def run_comments():
        try:
            current_artifact_root, _ = resolved_paths()
            document_number = validateFederalRegisterNumber(
                request.form.get("fr_document_number", "")
            )
            workers = _optional_int("workers")
            threshold = _optional_float("review_threshold", 0, 1)
            current = loadConfig(config_path)
            run_id = create_run_id()
            job = jobs.submit(
                "comments",
                run_id,
                collectAndAnalyzeComments,
                frDocumentNumber=document_number,
                downloadRoot=str(current_artifact_root),
                spreadsheetUrl=request.form.get("sheet_url", "").strip() or None,
                comparisonModels=(
                    []
                    if "no_comparison" in request.form
                    else current["ai"]["comparison"]["models"]
                    if current["ai"]["comparison"]["enabled"]
                    else []
                ),
                reviewThreshold=(
                    threshold
                    if threshold is not None
                    else current["pipeline"]["review_threshold"]
                ),
                maxWorkers=(
                    workers
                    if workers is not None
                    else current["pipeline"]["max_workers"]
                ),
                runId=run_id,
                configPath=str(config_path),
                useConfiguredSheet="configured_sheet" in request.form,
                noticeTitle=request.form.get("notice_title", "").strip() or None,
            )
        except ValueError as exc:
            flash(str(exc), "error")
            return redirect(url_for("dashboard"))
        flash(f"Comment analysis for {document_number} was queued.", "success")
        return redirect(url_for("job_detail", job_id=job["job_id"]))

    @app.post("/run/workflow")
    def run_workflow():
        try:
            current_artifact_root, _ = resolved_paths()
            analyze_top = _optional_int("analyze_top")
            if analyze_top is None:
                raise ValueError("analyze top is required")
            max_notices = _optional_int("max_notices")
            run_id = create_run_id()
            job = jobs.submit(
                "workflow",
                run_id,
                runConnectedWorkflow,
                analyzeTop=analyze_top,
                configPath=str(config_path),
                downloadRoot=str(current_artifact_root),
                maxNotices=max_notices,
                skipNoticeAI="skip_notice_ai" in request.form,
                useConfiguredSheet="configured_sheet" in request.form,
                runId=run_id,
            )
        except ValueError as exc:
            flash(str(exc), "error")
            return redirect(url_for("dashboard"))
        flash("Connected notice-to-comment workflow was queued.", "success")
        return redirect(url_for("job_detail", job_id=job["job_id"]))

    @app.get("/jobs/<job_id>")
    def job_detail(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            abort(404)
        return render_template("job.html", job=job)

    @app.get("/api/jobs/<job_id>")
    def job_api(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            return jsonify({"error": "job not found"}), 404
        return jsonify(job)

    @app.route("/settings", methods=["GET", "POST"])
    def settings():
        if request.method == "POST":
            yaml_text = request.form.get("yaml_text", "")
            try:
                _, current_log_dir = resolved_paths()
                saveConfigText(
                    config_path,
                    yaml_text,
                    actor="flask_web",
                    auditPath=current_log_dir / "config_changes.jsonl",
                )
                # Reloading validates the exact persisted representation.
                loadConfig(config_path)
            except ValueError as exc:
                flash(str(exc), "error")
                return render_template(
                    "settings.html",
                    yaml_text=yaml_text,
                    config_path=str(config_path),
                ), 400
            flash(
                "Configuration saved. New jobs use the updated workflow settings; "
                "host, port, and web worker changes apply after restart.",
                "success",
            )
            return redirect(url_for("settings"))
        yaml_text = config_path.read_text(encoding="utf-8")
        return render_template(
            "settings.html",
            yaml_text=yaml_text,
            config_path=str(config_path),
        )

    @app.get("/runs/<run_id>")
    def run_detail(run_id: str):
        try:
            validate_run_id(run_id)
        except ValueError:
            abort(404)
        current_artifact_root, _ = resolved_paths()
        runs_root = (current_artifact_root / "runs").resolve()
        run_dir = (runs_root / run_id).resolve()
        try:
            run_dir.relative_to(runs_root)
        except ValueError:
            abort(404)
        path = run_dir / "run_manifest.json"
        if not path.is_file():
            path = run_dir / "connected_workflow.json"
        payload = _load_json(path)
        if payload is None:
            abort(404)
        return render_template(
            "run.html",
            run=payload,
            run_id=run_id,
            manifest_path=str(path),
        )

    @app.get("/notices/<document_number>")
    def notice_detail(document_number: str):
        try:
            normalized = validateFederalRegisterNumber(document_number)
        except ValueError:
            abort(404)
        current_artifact_root, _ = resolved_paths()
        path = (
            current_artifact_root
            / "notices"
            / normalized
            / "analysis.json"
        )
        payload = _load_json(path)
        if payload is None:
            abort(404)
        return render_template(
            "notice.html",
            notice=payload,
            document_number=normalized,
            analysis_path=str(path),
        )

    @app.errorhandler(400)
    def bad_request(error):
        return render_template(
            "error.html",
            status=400,
            message=getattr(error, "description", "Invalid request."),
        ), 400

    @app.errorhandler(404)
    def not_found(error):
        return render_template(
            "error.html",
            status=404,
            message="That page or artifact was not found.",
        ), 404

    return app


def main() -> int:
    config_path = os.getenv("GRAIL_CONFIG", "config.yaml")
    config = loadConfig(config_path)
    configure_logging(
        resolveConfigPath(
            config_path,
            config["application"]["log_dir"],
        ),
        config["pipeline"]["log_level"],
    )
    app = createApp(config_path)
    app.run(
        host=config["web"]["host"],
        port=config["web"]["port"],
        debug=config["web"]["debug"],
        use_reloader=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
