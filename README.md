# Music Genre Classification

A machine learning project that classifies music into genres using the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification). Audio files are analyzed via acoustic feature extraction, and a Support Vector Machine predicts one of 10 genres.

## Business Understanding

This is a **supervised multi-class classification** problem that uses 57 numerical audio features (MFCCs, chroma, spectral characteristics, tempo, etc.) extracted via `librosa` from the GTZAN dataset to predict which of 10 musical genres (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock) a given audio clip belongs to. It has practical applications in music streaming recommendation, automatic catalog organization, playlist generation, and content moderation.

## Model Findings — SVM (SVC, RBF Kernel, C=15)

A `GridSearchCV` with 5-fold cross-validation compared four classifiers — SVC (RBF kernel, C=15), Random Forest, K-Nearest Neighbors, and Logistic Regression — all behind a `StandardScaler`. The **SVM (SVC)** was selected as the best model, achieving an overall **validation accuracy of 92.19%** on the held-out test set (999 samples).

### Per-Genre Performance

| Genre | Precision | Recall | F1-Score |
|---|---|---|---|
| **metal** (best) | 0.96 | 0.97 | 0.96 |
| **hiphop** | 0.94 | 0.95 | 0.95 |
| **pop** | 0.94 | 0.94 | 0.94 |
| **blues** | 0.91 | 0.97 | 0.94 |
| **reggae** | 0.95 | 0.92 | 0.93 |
| **jazz** | 0.93 | 0.93 | 0.93 |
| **classical** | 0.88 | 0.98 | 0.93 |
| **disco** | 0.92 | 0.89 | 0.90 |
| **country** | 0.86 | 0.87 | 0.86 |
| **rock** (weakest) | 0.94 | **0.79** | 0.86 |

### Key Takeaways

- The model is **strongest** at classifying **metal**, **hiphop**, and **pop** — genres with distinctive audio signatures (e.g., metal's high spectral energy, hiphop's percussive patterns, pop's tonal consistency).
- The model struggles most with **rock** and **country**. Rock has high precision (0.94) but notably low recall (0.79), meaning the model misses ~21% of actual rock clips, likely confusing them with nearby genres like country, blues, or metal.
- **Country** shows the lowest precision (0.86), meaning other genres are occasionally mislabeled as country.
- **Classical** has very high recall (0.98) but somewhat lower precision (0.88), suggesting the model rarely misses a classical clip but occasionally misclassifies something else as classical.
- The **macro average F1** (0.92) is close to the overall accuracy, indicating the model is well-balanced across all 10 classes.
- The main area of confusion is the **rock/country/blues cluster** — genres that share similar instrumentation and tempo characteristics in the real world.

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

The notebook (`capstone_project.ipynb`) explores several approaches on the 3-second feature set:

**Standard models** — 5-fold cross-validated grid search over SVC, Random Forest, K-Nearest Neighbors, and Logistic Regression. SVC (RBF, C=15) comes out best with ~92% validation accuracy.

**Neural network** — A dense MLP (Keras) is trained on the same features; validation accuracy reaches ~88–91% depending on training.

**Best SVM** — A refined grid search over SVC hyperparameters (C, gamma, class_weight) selects `C=15`, `gamma='scale'`, `class_weight=None`, yielding ~92% test accuracy.

The production pipeline in `music_genre_classifier.py` is:

```
StandardScaler → SVC(kernel='rbf', C=15)
```

This achieves ~91–92% accuracy on a held-out 20% test split.

## Project Structure

```
capstone_project.ipynb      # EDA, model comparison (SVC, RF, KNN, LogReg, MLP), hyperparameter tuning, feature extraction
music_genre_classifier.py   # MusicGenreClassifier class wrapping the final SVC pipeline
streamlit/
  Classifier.py             # Streamlit app: upload & classify
  pages/
    1_Model_Stats.py        # Model Stats page: confusion matrix, classification report
  uploads/                  # Files uploaded via the web app
data/
  features_3_sec.csv        # Pre-extracted 3-second features
  features_30_sec.csv       # Pre-extracted 30-second features
  test_songs/               # Sample audio files for inference testing
```

## Streamlit App

The app lives under `streamlit/`. You can upload one or more `.wav`, `.mp3`, or `.ogg` files; for each file you can set a **portion** (0.01–1.0) of the track to use for feature extraction (default 0.25). Use **Classify** per row or **Classify All** to run the pipeline. A **Model Stats** page shows accuracy, precision, recall, F1, confusion matrix, and per-genre classification report.

Run from the project root:

```bash
streamlit run streamlit/Classifier.py
```

## MusicGenreClassifier

The `MusicGenreClassifier` class in `music_genre_classifier.py` loads the GTZAN CSV data, trains the pipeline, and exposes:

- `predict(file_path, portion=0.25)` – extract features from a portion of the audio (by duration), aggregate over 3-second segments, and return the predicted genre
- `classification_report()` – per-class precision/recall/F1
- `confusion_matrix()` – confusion matrix

Test-set accuracy and other metrics are available via the Streamlit **Model Stats** page or by calling `classification_report()` / `confusion_matrix()`.

```python
from music_genre_classifier import MusicGenreClassifier

clf = MusicGenreClassifier()                   # trains on 3-sec features by default
print(clf.classification_report())              # per-class metrics
print(clf.predict('my_song.wav'))               # e.g. "Country"
print(clf.predict('my_song.wav', portion=0.5))  # use first half of track
```

## Dependencies

- Python 3.x
- `librosa` – audio feature extraction
- `scikit-learn` – modeling pipeline (SVC, StandardScaler)
- `pandas`, `numpy` – data handling
- `streamlit` – web app
- `matplotlib` – visualization (notebook and Model Stats page)
- `keras` – neural network experiments in the notebook only
