# core loop
import io
import logging

import flask

from src.collect_notices.collect import collectNotices
from src.collect_comments.fetch_comments import processComments
from src.services.config import loadNoticeConfig

app = flask.Flask(__name__, template_folder="../templates")


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


@app.route("/comments")
def comments():
    return flask.render_template("comments.html")


@app.route("/settings")
def settings():
    return flask.render_template("settings.html")


def main() -> None:
    app.run()


if __name__ == "__main__":
    main()
