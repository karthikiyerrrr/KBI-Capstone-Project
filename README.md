# Music Genre Classification

A machine learning project that classifies music into genres using the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification). Audio files are analyzed via acoustic feature extraction, and a Support Vector Machine predicts one of 10 genres.

## Dataset

The GTZAN dataset contains ~1,000 samples per genre across 10 genres:

**blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock**

Pre-extracted features are provided in two CSV files: `features_3_sec.csv` (3-second clips) and `features_30_sec.csv` (30-second clips), each with 57 acoustic features per sample.

## Features

Each audio sample is represented by the mean and variance of:

- **Chroma STFT** – pitch class energy distribution
- **RMS** – overall loudness/energy
- **Spectral Centroid** – brightness of sound
- **Spectral Bandwidth** – range of frequencies present
- **Spectral Rolloff** – frequency below which most energy is concentrated
- **Zero Crossing Rate** – noisiness / percussiveness
- **Harmony & Percussive** – separated harmonic and percussive components (via HPSS)
- **20 MFCCs** – timbral texture coefficients

Plus a single **Tempo** value — 57 features total.

## Model

The notebook (`capstone_project.ipynb`) compares SVC and Random Forest classifiers via 5-fold cross-validated grid search on the 3-second feature set:

| Model | Best Params | Test Accuracy |
|---|---|---|
| SVC | kernel=rbf, C=10 | **93%** |
| Random Forest | (various) | lower |

A second grid search refined the SVC further with PCA dimensionality reduction (retaining 95% of variance) and tested C values of 10, 15, and 25. The final pipeline is:

```
StandardScaler → PCA(n_components=0.95) → SVC(kernel='rbf', C=15)
```

This achieves ~91–93% accuracy on a held-out 20% test split (1,998 samples).

## Project Structure

```
capstone_project.ipynb      # EDA, model selection, hyperparameter tuning, feature extraction demo
music_genre_classifier.py   # MusicGenreClassifier class wrapping the final pipeline
Classifier.py               # Streamlit web app
data/
  features_3_sec.csv        # Pre-extracted 3-second features
  features_30_sec.csv       # Pre-extracted 30-second features
  test_songs/               # Sample audio files for inference testing
uploads/                    # Directory for files uploaded via the web app
```

## Streamlit App

`Classifier.py` provides a simple web interface. Upload a `.wav`, `.mp3`, or `.ogg` file and the app extracts features using `librosa` and returns the predicted genre.

To run:

```bash
streamlit run Classifier.py
```

## MusicGenreClassifier

The `MusicGenreClassifier` class in `music_genre_classifier.py` loads the GTZAN CSV data, trains the pipeline, and exposes:

- `predict(file_path)` – extract features from an audio file and return the predicted genre
- `accuracy()` – test set accuracy
- `classification_report()` – per-class precision/recall/F1
- `confusion_matrix()` – confusion matrix

```python
from music_genre_classifier import MusicGenreClassifier

clf = MusicGenreClassifier()          # trains on 3-sec features by default
print(clf.accuracy())                 # ~0.91
print(clf.predict('my_song.wav'))     # e.g. "Country"
```

## Dependencies

- Python 3.x
- `librosa` – audio feature extraction
- `scikit-learn` – modeling pipeline (SVC, PCA, StandardScaler)
- `pandas`, `numpy` – data handling
- `streamlit` – web app
- `matplotlib` – visualization (notebook only)
