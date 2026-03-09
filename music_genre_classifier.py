import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

data_3_sec = pd.read_csv('data/features_3_sec.csv')
data_30_sec = pd.read_csv('data/features_30_sec.csv')

class MusicGenreClassifier:
    def __init__(self, use_30_sec=False):
        self.svc_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)),
            ('svc', SVC(kernel='rbf', C=15))
        ])
        self.X_train, self.X_test, self.y_train, self.y_test = self._load_data(use_30_sec)
        self.svc_pipe.fit(self.X_train, self.y_train)

    def _load_data(self, use_30_sec=False):
        self.le = LabelEncoder()
        if use_30_sec:
            self.data = pd.read_csv('data/features_30_sec.csv')
        else:
            self.data = pd.read_csv('data/features_3_sec.csv')
        
        self.X = self.data.drop(columns=['filename', 'length', 'label'])
        self.y = self.le.fit_transform(self.data['label'])
        return train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def _extract_features(self, file_path):
        y, sr = librosa.load(file_path, mono=True, duration=30)
        
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        harmony, perceptr = librosa.effects.hpss(y)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

        features = {
            'chroma_stft_mean': np.mean(chroma_stft),
            'chroma_stft_var': np.var(chroma_stft),
            'rms_mean': np.mean(rms),
            'rms_var': np.var(rms),
            'spectral_centroid_mean': np.mean(spec_cent),
            'spectral_centroid_var': np.var(spec_cent),
            'spectral_bandwidth_mean': np.mean(spec_bw),
            'spectral_bandwidth_var': np.var(spec_bw),
            'rolloff_mean': np.mean(rolloff),
            'rolloff_var': np.var(rolloff),
            'zero_crossing_rate_mean': np.mean(zcr),
            'zero_crossing_rate_var': np.var(zcr),
            'harmony_mean': np.mean(harmony),
            'harmony_var': np.var(harmony),
            'perceptr_mean': np.mean(perceptr),
            'perceptr_var': np.var(perceptr),
            'tempo': float(np.atleast_1d(tempo).flat[0]),
        }

        for i in range(1, 21):
            features[f'mfcc{i}_mean'] = np.mean(mfccs[i-1])
            features[f'mfcc{i}_var'] = np.var(mfccs[i-1])

        return features

    def predict(self, file_path):
        features = self._extract_features(file_path)
        features_df = pd.DataFrame([features])
        prediction = self.svc_pipe.predict(features_df)
        genre = self.le.inverse_transform(prediction)[0]
        return str(genre).capitalize()

    def accuracy(self):
        return self.svc_pipe.score(self.X_test, self.y_test)

    def classification_report(self):
        y_pred = self.svc_pipe.predict(self.X_test)
        return classification_report(self.y_test, y_pred, target_names=self.le.classes_)

    def confusion_matrix(self):
        y_pred = self.svc_pipe.predict(self.X_test)
        return confusion_matrix(self.y_test, y_pred)