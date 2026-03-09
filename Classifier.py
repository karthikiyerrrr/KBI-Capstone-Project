import streamlit as st
from music_genre_classifier import MusicGenreClassifier

if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'last_processed' not in st.session_state:
    st.session_state.last_processed = None

music_genre_classifier = MusicGenreClassifier()

st.sidebar.header('Upload')
uploaded_file = st.sidebar.file_uploader('Choose a music file', type=['wav', 'mp3', 'ogg'])

if uploaded_file is not None and uploaded_file.name != st.session_state.last_processed:
    file_path = 'uploads/' + uploaded_file.name
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    prediction_genre = music_genre_classifier.predict(file_path)
    st.session_state.predictions.append({
        'File Name': uploaded_file.name,
        'Predicted Genre': prediction_genre,
    })
    st.session_state.last_processed = uploaded_file.name

st.title('Music Genre Classification')

if not st.session_state.predictions:
    st.write('No files uploaded.')
else:
    st.dataframe(
        st.session_state.predictions,
        use_container_width=True,
        hide_index=True,
    )
