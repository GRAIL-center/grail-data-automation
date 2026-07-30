# core loop
import io
import json
import logging
import queue
import threading
import uuid
from pathlib import Path

import flask
import yaml

from src.collect_comments.fetch_comments import processComments
from src.collect_notices.collect import collectNotices
from src.services.ai_client import reset_client
from src.services.config import CONFIG_PATH, loadNoticeConfig

app = flask.Flask(__name__, template_folder="../templates")
commentRuns: dict[str, dict] = {}
commentRunsLock = threading.Lock()


class LiveLogHandler(logging.Handler):
    def __init__(self, logQueue: queue.Queue):
        super().__init__()
        self.logQueue = logQueue

    def emit(self, record: logging.LogRecord) -> None:
        self.logQueue.put(self.format(record))


def splitTerms(value: str) -> list[str]:
    return [term.strip() for term in value.splitlines() if term.strip()]


def runNoticeCollection(settings: dict, terms: dict) -> tuple[str, str]:
    logStream = io.StringIO()
    handler = logging.StreamHandler(logStream)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    collectionLogger = logging.getLogger("src.collect_notices.collect")
    previousLevel = collectionLogger.level
    collectionLogger.setLevel(logging.INFO)
    collectionLogger.addHandler(handler)

    try:
        collectNotices(settings, terms)
        status = "Notice collection finished."
    except Exception:
        collectionLogger.exception("Notice collection failed")
        status = "Notice collection failed. Review the log for details."
    finally:
        collectionLogger.removeHandler(handler)
        collectionLogger.setLevel(previousLevel)
        handler.close()

    return logStream.getvalue(), status


@app.route("/")
def index():
    return flask.render_template("index.html")


@app.route("/notices", methods=["GET", "POST"])
def notices():
    notice_settings, notice_terms = loadNoticeConfig()
    notice_settings = notice_settings or {}
    notice_terms = notice_terms or {}
    notice_log = ""
    run_status = ""

    if flask.request.method == "POST":
        notice_settings = {
            "max_notices": int(flask.request.form["max_notices"]),
            "docket_type": flask.request.form["docket_type"],
            "order": flask.request.form["order"],
            "start_date": flask.request.form["start_date"],
        }
        notice_terms = {
            "default_terms": splitTerms(flask.request.form.get("default_terms", "")),
            "search_terms": splitTerms(flask.request.form.get("search_terms", "")),
        }
        notice_log, run_status = runNoticeCollection(notice_settings, notice_terms)

    return flask.render_template(
        "notices.html",
        notice_settings=notice_settings,
        notice_terms=notice_terms,
        notice_log=notice_log,
        run_status=run_status,
    )


def runCommentCollection(run: dict, frNumber: str, spreadsheetUrl: str | None) -> None:
    logQueue = run["log_queue"]
    logHandler = LiveLogHandler(logQueue)
    logHandler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))

    collectionLogger = logging.getLogger("src.collect_comments")
    previousLevel = collectionLogger.level
    collectionLogger.setLevel(logging.INFO)
    collectionLogger.addHandler(logHandler)

    try:
        collectionLogger.info("Starting comment collection for %s", frNumber)
        commentCount = processComments(frNumber, spreadsheetUrl)
        run["status"] = "completed"
        run["message"] = f"Processed {commentCount} comment(s)."
        collectionLogger.info(run["message"])
    except Exception:
        run["status"] = "failed"
        run["message"] = "Comment collection failed. Review the log for details."
        collectionLogger.exception("Comment collection failed")
    finally:
        collectionLogger.removeHandler(logHandler)
        collectionLogger.setLevel(previousLevel)
        logHandler.close()
        logQueue.put(None)


@app.route("/comments")
def comments():
    return flask.render_template("comments.html")


@app.route("/comments/run", methods=["POST"])
def startComments():
    frNumber = flask.request.form.get("fr_number", "").strip()
    spreadsheetUrl = flask.request.form.get("spreadsheet_url", "").strip() or None

    if not frNumber:
        return flask.jsonify({"error": "A Federal Register document number is required."}), 400

    with commentRunsLock:
        if any(run["status"] == "running" for run in commentRuns.values()):
            return flask.jsonify({"error": "A comment collection is already running."}), 409

        runId = uuid.uuid4().hex
        run = {
            "log_queue": queue.Queue(),
            "status": "running",
            "message": "Comment collection is running.",
        }
        commentRuns[runId] = run

    worker = threading.Thread(
        target=runCommentCollection,
        args=(run, frNumber, spreadsheetUrl),
        daemon=True,
    )
    worker.start()

    return flask.jsonify({"run_id": runId})


@app.route("/comments/logs/<runId>")
def streamCommentLogs(runId: str):
    run = commentRuns.get(runId)
    if run is None:
        flask.abort(404)

    def eventStream():
        while True:
            try:
                message = run["log_queue"].get(timeout=15)
            except queue.Empty:
                yield ": keep-alive\n\n"
                continue

            if message is None:
                completion = json.dumps(
                    {"status": run["status"], "message": run["message"]}
                )
                yield f"event: complete\ndata: {completion}\n\n"
                break

            yield f"data: {json.dumps(message)}\n\n"

    return flask.Response(
        flask.stream_with_context(eventStream()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/settings", methods=["GET", "POST"])
def settings():
    configPath = Path(CONFIG_PATH)
    configText = configPath.read_text(encoding="utf-8")
    saveMessage = ""
    saveError = ""

    if flask.request.method == "POST":
        configText = flask.request.form.get("config_yaml", "")

        try:
            configData = yaml.safe_load(configText)
        except yaml.YAMLError as error:
            saveError = f"Config was not saved: invalid YAML ({error})."
        else:
            if not isinstance(configData, dict):
                saveError = "Config was not saved: the top-level YAML value must be a mapping."
            else:
                temporaryPath = configPath.with_name(f"{configPath.name}.tmp")
                try:
                    temporaryPath.write_text(configText, encoding="utf-8")
                    temporaryPath.replace(configPath)
                    reset_client()
                    saveMessage = "config.yaml saved successfully."
                except OSError as error:
                    saveError = f"Config was not saved: {error}."

    return flask.render_template(
        "settings.html",
        config_text=configText,
        save_message=saveMessage,
        save_error=saveError,
    )


def main() -> None:
    app.run()


if __name__ == "__main__":
    main()
