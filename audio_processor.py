import numpy as np
import torch
import librosa
import soundfile as sf

DEVICE = torch.device("cpu")

def extract_covarep_like_features(audio_path: str, sr: int = 16000) -> np.ndarray:
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        features_list = []

        # MFCCs (13)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features_list.append(mfcc.T)

        # Delta MFCCs
        delta_mfcc = librosa.feature.delta(mfcc)
        features_list.append(delta_mfcc.T)

        # Spectral features
        features_list.append(librosa.feature.spectral_centroid(y=y, sr=sr).T)
        features_list.append(librosa.feature.spectral_bandwidth(y=y, sr=sr).T)
        features_list.append(librosa.feature.spectral_rolloff(y=y, sr=sr).T)
        features_list.append(librosa.feature.spectral_contrast(y=y, sr=sr).T)

        # Pitch
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr, frame_length=2048)
        features_list.append(np.nan_to_num(f0).reshape(-1, 1))

        # RMS & ZCR
        features_list.append(librosa.feature.rms(y=y).T)
        features_list.append(librosa.feature.zero_crossing_rate(y).T)

        min_len  = min(f.shape[0] for f in features_list)
        features = np.hstack([f[:min_len] for f in features_list])
        return np.nan_to_num(features).astype(np.float32)

    except Exception as e:
        print(f"Audio feature extraction error: {e}")
        return None


def extract_raw_features_for_shap(audio_path: str, audio_config: dict) -> np.ndarray:
    """
    Extract aggregated (mean across frames) raw features for SHAP.
    Returns shape (1, n_features) — unscaled, one vector per recording.
    """
    features = extract_covarep_like_features(audio_path)
    if features is None:
        return None

    expected_dim = audio_config["input_dim"]
    actual_dim   = features.shape[1]

    if actual_dim != expected_dim:
        if actual_dim < expected_dim:
            pad      = np.zeros((features.shape[0], expected_dim - actual_dim))
            features = np.hstack([features, pad])
        else:
            features = features[:, :expected_dim]

    # Aggregate across time → single vector (mean per feature)
    return features.mean(axis=0, keepdims=True).astype(np.float32)


def process_audio_for_model(audio_path, audio_scaler, audio_model, audio_config):
    features = extract_covarep_like_features(audio_path)
    if features is None:
        return None, None

    expected_dim = audio_config["input_dim"]
    actual_dim   = features.shape[1]

    if actual_dim != expected_dim:
        if actual_dim < expected_dim:
            pad      = np.zeros((features.shape[0], expected_dim - actual_dim))
            features = np.hstack([features, pad])
        else:
            features = features[:, :expected_dim]

    try:
        features = audio_scaler.transform(features)
    except:
        pass

    # Pad/truncate to 512 frames
    if len(features) > 512:
        features = features[:512]
    else:
        pad      = np.zeros((512 - len(features), features.shape[1]))
        features = np.vstack([features, pad])

    tensor = torch.tensor(features[np.newaxis], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        embedding  = audio_model.get_embeddings(tensor)
        logits     = audio_model(tensor)
        probs      = torch.softmax(logits, dim=1)
        confidence = probs[0][1].item()

    return embedding.cpu().numpy(), confidence


def transcribe_audio(audio_path: str) -> str:
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""