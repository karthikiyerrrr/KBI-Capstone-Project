import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from music_genre_classifier import MusicGenreClassifier

music_genre_classifier = MusicGenreClassifier()

if st.button("← Back"):
    st.switch_page("Classifier.py")

st.title("Model Stats")

# Compute once for pills, confusion matrix, and classification report tables
classes = music_genre_classifier.le.classes_
y_pred = music_genre_classifier.svc_pipe.predict(music_genre_classifier.X_test)
report_dict = classification_report(
    music_genre_classifier.y_test,
    y_pred,
    target_names=classes,
    output_dict=True,
)
cm = confusion_matrix(music_genre_classifier.y_test, y_pred)

# Pillboxes: accuracy + overall (macro) precision, recall, f1
acc = f"{report_dict['accuracy']:.2f}"
macro = report_dict["macro avg"]
prec = f"{macro['precision']:.2f}"
rec = f"{macro['recall']:.2f}"
f1 = f"{macro['f1-score']:.2f}"
pill_style = (
    "display: inline-block; padding: 0.4rem 0.85rem; border-radius: 9999px;"
    " background: var(--secondary-background-color, #262730); margin: 0.25rem 0.45rem 0.25rem 0;"
    " font-size: 1.05rem; font-weight: 600;"
)
code_style = "font-family: ui-monospace, monospace; font-size: 0.9em;"
st.markdown(
    f"""
    <div style="display: flex; flex-wrap: wrap; align-items: center; margin: 0.5rem 0;">
        <div style="{pill_style}">Accuracy: <code style="{code_style}">{acc}</code></div>
        <div style="{pill_style}">Precision: <code style="{code_style}">{prec}</code></div>
        <div style="{pill_style}">Recall: <code style="{code_style}">{rec}</code></div>
        <div style="{pill_style}">F1: <code style="{code_style}">{f1}</code></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Theme colors (Streamlit palette); fallback for dark theme
try:
    bg_hex = st.get_option("theme.backgroundColor") or "#0e1117"
    text_hex = st.get_option("theme.textColor") or "#fafafa"
except Exception:
    bg_hex = "#0e1117"
    text_hex = "#fafafa"


def lighten_hex(hex_str, amount=18):
    """Return a slightly lighter hex color (add amount to R,G,B, cap 255)."""
    hex_str = hex_str.lstrip("#")
    r, g, b = (int(hex_str[i : i + 2], 16) for i in (0, 2, 4))
    r = min(255, r + amount)
    g = min(255, g + amount)
    b = min(255, b + amount)
    return f"#{r:02x}{g:02x}{b:02x}"


zero_hex = lighten_hex(bg_hex)
# Colormap: 0 -> slightly lighter than bg; max -> darker accent (use primary or a blue)
try:
    primary_hex = st.get_option("theme.primaryColor") or "#ff4b4b"
except Exception:
    primary_hex = "#ff4b4b"
# Build a dark colormap: zero_hex -> darker shades toward primary
cmap = mcolors.LinearSegmentedColormap.from_list(
    "confusion", [zero_hex, primary_hex], N=256
)

fig, ax = plt.subplots()
fig.patch.set_facecolor(bg_hex)
ax.set_facecolor(bg_hex)
im = ax.imshow(
    cm,
    interpolation="nearest",
    cmap=cmap,
    vmin=0,
    vmax=cm.max() if cm.size > 0 else 1,
)
cbar = fig.colorbar(im, ax=ax)
cbar.ax.yaxis.set_tick_params(color=text_hex)
plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=text_hex)
ax.set(
    xticks=range(len(classes)),
    yticks=range(len(classes)),
    xticklabels=classes,
    yticklabels=classes,
    xlabel="Predicted",
    ylabel="True",
)
ax.tick_params(axis="both", colors=text_hex)
ax.xaxis.label.set_color(text_hex)
ax.yaxis.label.set_color(text_hex)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
for i in range(len(classes)):
    for j in range(len(classes)):
        val = cm[i, j]
        text_color = "white" if (val > cm.max() / 2 and cm.max() > 0) else text_hex
        ax.text(j, i, int(val), ha="center", va="center", color=text_color)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("Classification Report")

def _table_html(header_cells, rows, primary_hex):
    """Render an HTML table with header row styled with primary color."""
    thead = "".join(f"<th>{c}</th>" for c in header_cells)
    tbody = ""
    for row in rows:
        tbody += "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
    return f"""
    <table style="border-collapse: collapse; width: 100%; margin: 0.5rem 0;">
    <thead><tr style="background: {primary_hex}; color: white;">{thead}</tr></thead>
    <tbody>{tbody}</tbody>
    </table>
    <style> table th, table td {{ padding: 0.4rem 0.75rem; text-align: left; border: 1px solid #333; }} </style>
    """

# Per-category table (exclude accuracy, macro avg, weighted avg)
class_keys = [k for k in report_dict if k not in ("accuracy", "macro avg", "weighted avg")]
if class_keys:
    header = ["Genre", "Precision", "Recall", "F1-score", "Support"]
    rows = [
        [k, f"{report_dict[k]['precision']:.2f}", f"{report_dict[k]['recall']:.2f}",
         f"{report_dict[k]['f1-score']:.2f}", str(int(report_dict[k]['support']))]
        for k in class_keys
    ]
    st.markdown(_table_html(header, rows, primary_hex), unsafe_allow_html=True)

st.divider()
header_avg = ["", "Precision", "Recall", "F1-score", "Support"]
rows_avg = [
    ["Macro Avg", f"{report_dict['macro avg']['precision']:.2f}", f"{report_dict['macro avg']['recall']:.2f}",
     f"{report_dict['macro avg']['f1-score']:.2f}", str(int(report_dict['macro avg']['support']))],
    ["Weighted Avg", f"{report_dict['weighted avg']['precision']:.2f}", f"{report_dict['weighted avg']['recall']:.2f}",
     f"{report_dict['weighted avg']['f1-score']:.2f}", str(int(report_dict['weighted avg']['support']))],
]
st.markdown(_table_html(header_avg, rows_avg, primary_hex), unsafe_allow_html=True)
