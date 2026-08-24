# REALM dashboard

This is the REALM experiment dashboard, colocated with the benchmark repository.

From the repository root, install the host-side environment:

```bash
uv sync
```

Then run the dashboard against the repository's default `logs/` directory:

```bash
uv run streamlit run tooling/realm_viewer/dashboard.py
```

To use another logs directory:

```bash
REALM_LOGS=/path/to/REALM/logs uv run streamlit run tooling/realm_viewer/dashboard.py
```

Streamlit serves the dashboard at <http://localhost:8501> by default.
