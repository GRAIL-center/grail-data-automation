# Streamlit interface

The Streamlit interface is a standalone alternative to the Flask UI. It uses
existing collector functions and does not replace or modify Flask routes.

## Install

```sh
python -m pip install -r requirements.txt
```

## Run

From the project root:

```sh
streamlit run streamlit_app.py
```

## Features

- Notice collection with editable settings and search terms
- Comment collection with live log updates
- Artifact browser for `data/<FR number>/<comment ID>/`
- Validated `config.yaml` editor that refreshes future AI-client use after save

API keys and Google Sheet URLs remain in `.env`; they are never displayed in
the Streamlit interface.
