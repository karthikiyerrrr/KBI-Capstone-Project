import sys
from pathlib import Path

import streamlit as st

# Allow importing music_genre_classifier from project root when run as streamlit/Classifier.py
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from music_genre_classifier import MusicGenreClassifier

UPLOADS_DIR = _root / "streamlit" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

GENRE_ICONS = {
    "blues": "🎷",
    "classical": "🎻",
    "country": "🎸",
    "disco": "💿",
    "hiphop": "🎤",
    "jazz": "🎺",
    "metal": "🤘",
    "pop": "⭐",
    "reggae": "🌴",
    "rock": "🎸",
}


def genre_display(genre_str):
    """Return icon + genre for display, or genre_str if no icon."""
    if not genre_str or genre_str in ("—", "Error", "N/A (file missing)"):
        return genre_str
    icon = GENRE_ICONS.get(genre_str.lower())
    return f"{icon} {genre_str}" if icon else genre_str

if "predictions" not in st.session_state:
    st.session_state.predictions = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

music_genre_classifier = MusicGenreClassifier()

st.sidebar.header("Upload")
uploaded_files = st.sidebar.file_uploader(
    "Choose music file(s)",
    type=["wav", "mp3", "ogg"],
    key=f"upload_{st.session_state.uploader_key}",
    accept_multiple_files=True,
)

if uploaded_files:
    for f in uploaded_files:
        file_path = UPLOADS_DIR / f.name
        file_path.write_bytes(f.getbuffer())
        st.session_state.predictions.append({
            "File Name": f.name,
            "Predicted Genre": "—",
            "portion": 0.25,
        })
    st.session_state.uploader_key += 1
    st.rerun()

st.title("Music Genre Classification")

predictions = st.session_state.predictions
if not predictions:
    st.write("No files uploaded.")
else:
    for row in predictions:
        if "portion" not in row:
            row["portion"] = 0.25

    try:
        primary_hex = st.get_option("theme.primaryColor") or "#ff4b4b"
    except Exception:
        primary_hex = "#ff4b4b"

    # Match Streamlit columns ratio [2, 1.5, 2, 1] (total 6.5) so header aligns with rows
    total = 2 + 1.5 + 2 + 1
    w1, w2, w3, w4 = 100 * 2 / total, 100 * 1.5 / total, 100 * 2 / total, 100 * 1 / total
    # Table header (primary hex; fixed column widths to match st.columns)
    header_html = f"""
    <table style="border-collapse: collapse; width: 100%; margin: 0.5rem 0; table-layout: fixed;">
    <thead><tr style="background: {primary_hex}; color: white;">
    <th style="width: {w1}%; padding: 0.4rem 0.75rem; text-align: left; border: 1px solid #333; vertical-align: middle;">File Name</th>
    <th style="width: {w2}%; padding: 0.4rem 0.75rem; text-align: left; border: 1px solid #333; vertical-align: middle;">Portion</th>
    <th style="width: {w3}%; padding: 0.4rem 0.75rem; text-align: left; border: 1px solid #333; vertical-align: middle;">Predicted Genre</th>
    <th style="width: {w4}%; padding: 0.4rem 0.75rem; text-align: left; border: 1px solid #333; vertical-align: middle;">Classify</th>
    </tr></thead></table>
    <style>
    div[data-testid="stHorizontalBlock"] {{ display: flex; align-items: center; }}
    </style>
    """
    st.markdown(header_html, unsafe_allow_html=True)

    # Data rows (columns align with header; vertical centering via CSS above)
    for idx, row in enumerate(predictions):
        cols = st.columns([2, 1.5, 2, 1])
        with cols[0]:
            st.text(row["File Name"])
        with cols[1]:
            portion = st.slider(
                "Portion",
                min_value=0.01,
                max_value=1.0,
                value=row.get("portion", 0.25),
                step=0.05,
                key=f"portion_{idx}",
                label_visibility="collapsed",
            )
            row["portion"] = portion
        with cols[2]:
            st.text(genre_display(row["Predicted Genre"]))
        with cols[3]:
            if st.button("Classify", key=f"classify_{idx}"):
                path = UPLOADS_DIR / row["File Name"]
                if not path.exists():
                    predictions[idx]["Predicted Genre"] = "N/A (file missing)"
                else:
                    portion_val = st.session_state.get(f"portion_{idx}", row.get("portion", 0.25))
                    try:
                        genre = music_genre_classifier.predict(
                            str(path), portion=portion_val
                        )
                        predictions[idx]["Predicted Genre"] = (
                            genre if genre is not None else "—"
                        )
                    except Exception:
                        predictions[idx]["Predicted Genre"] = "Error"
                st.rerun()
    if len(predictions) >= 2:
        _, btn_col = st.columns([5, 1])
        with btn_col:
            if st.button("Classify All"):
                for idx, row in enumerate(predictions):
                    path = UPLOADS_DIR / row["File Name"]
                    if not path.exists():
                        predictions[idx]["Predicted Genre"] = "N/A (file missing)"
                        continue
                    portion_val = st.session_state.get(f"portion_{idx}", row.get("portion", 0.25))
                    try:
                        genre = music_genre_classifier.predict(
                            str(path), portion=portion_val
                        )
                        predictions[idx]["Predicted Genre"] = (
                            genre if genre is not None else "—"
                        )
                    except Exception:
                        predictions[idx]["Predicted Genre"] = "Error"
                st.rerun()

