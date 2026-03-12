import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv('data/features_3_sec.csv')

class MusicGenreClassifier:
    def __init__(self):
        self.svc_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='rbf', C=15))
        ])
        self.X_train, self.X_test, self.y_train, self.y_test = self._load_data()
        self.svc_pipe.fit(self.X_train, self.y_train)

    def _load_data(self):
        self.le = LabelEncoder()
        self.data = data
        self.X = self.data.drop(columns=['filename', 'length', 'label'])
        self.y = self.le.fit_transform(self.data['label'])
        return train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def _features_from_segment(self, y, sr):
        """Compute one feature dict from a single audio segment (e.g. 3 seconds)."""
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

    def _extract_features(self, file_path, portion=1.0, segment_duration_sec=3):
        """Load a portion of the song's duration, split into whole 3s segments (round down), return list of feature dicts."""
        total_duration = librosa.get_duration(path=file_path)
        duration_to_use = total_duration * portion
        num_segments = int(duration_to_use // segment_duration_sec)
        if num_segments == 0:
            return []
        load_duration = num_segments * segment_duration_sec
        y, sr = librosa.load(file_path, mono=True, duration=load_duration)
        segment_samples = segment_duration_sec * sr

        feature_list = []
        for i in range(num_segments):
            start = i * segment_samples
            end = start + segment_samples
            if end > len(y):
                break
            y_seg = y[start:end]
            feature_list.append(self._features_from_segment(y_seg, sr))
        return feature_list

    def predict(self, file_path, portion=0.25):
        feature_list = self._extract_features(file_path, portion=portion)
        if not feature_list:
            return None
        features_df = pd.DataFrame(feature_list)
        predictions = self.svc_pipe.predict(features_df)
        counts = {}
        for p in predictions:
            counts[p] = counts.get(p, 0) + 1
        majority_label = max(counts, key=counts.get)
        genre = self.le.inverse_transform([majority_label])[0]
        return str(genre).capitalize()

    def classification_report(self):
        y_pred = self.svc_pipe.predict(self.X_test)
        return classification_report(self.y_test, y_pred, target_names=self.le.classes_)

    def confusion_matrix(self):
        y_pred = self.svc_pipe.predict(self.X_test)
        return confusion_matrix(self.y_test, y_pred)