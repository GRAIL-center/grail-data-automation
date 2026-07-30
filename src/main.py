# core loop
import flask

from src.services.config import loadNoticeConfig
from src.collect_notices.collect import collectNotices

app = flask.Flask(__name__, template_folder="../templates")

@app.route("/")
def index():
    return flask.render_template("index.html")


@app.route("/notices")
def notices():
    notice_settings, notice_terms = loadNoticeConfig()
    return flask.render_template(
        "notices.html",
        notice_settings=notice_settings or {},
        notice_terms=notice_terms or {},
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
