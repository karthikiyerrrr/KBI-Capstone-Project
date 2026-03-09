import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import classification_report

from music_genre_classifier import MusicGenreClassifier

music_genre_classifier = MusicGenreClassifier()

if st.button('← Back'):
    st.switch_page('capstone_streamlit.py')

st.title('Model stats')

accuracy = music_genre_classifier.accuracy()
st.markdown(
    f"<p style='font-size: 3rem; font-weight: 700; margin: 0.5rem 0;'>Accuracy: {accuracy * 100:.1f}%</p>",
    unsafe_allow_html=True,
)

cm = music_genre_classifier.confusion_matrix()
classes = music_genre_classifier.le.classes_

fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
       xticklabels=classes, yticklabels=classes,
       xlabel='Predicted', ylabel='True')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

y_pred = music_genre_classifier.svc_pipe.predict(music_genre_classifier.X_test)
report_dict = classification_report(
    music_genre_classifier.y_test,
    y_pred,
    target_names=classes,
    output_dict=True,
)
report_df = pd.DataFrame(report_dict).T.reset_index()
report_df = report_df.rename(columns={'index': 'Genre'})
st.subheader('Classification report')
st.dataframe(report_df, use_container_width=True, hide_index=True)
