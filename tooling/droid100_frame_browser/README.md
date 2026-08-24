# DROID100 first-frame browser

Browse the `panel.jpg` / `panel.jpeg` render panels pulled into
`logs/droid100_first_frames/<run>/frames/<task>/`:

```sh
uv run streamlit run tooling/droid100_frame_browser/dashboard.py --server.port 8505
```

The dashboard supports run and status filters, task search, sequential navigation, a gallery view,
and local review verdicts with notes. Reviews are saved as
`logs/droid100_first_frames/<run>/frame_review.json`; rendered images are never modified.
