"""
audio_processor.py — MindCare AI
Audio feature extraction using opensmile eGeMAPS (clinical standard).

eGeMAPS = Extended Geneva Minimalistic Acoustic Parameter Set
  - 88 features validated for depression/emotion clinical research
  - Used in AVEC 2013-2019 depression detection challenges
  - Covers: frequency, energy, spectral, temporal, cepstral, voice quality

Falls back to librosa if opensmile is not installed.
"""

import numpy as np
import torch
import soundfile as sf
import os

DEVICE = torch.device("cpu")

# ── eGeMAPS feature names (88 features) ───────────────────────────
EGEMAPS_FEATURE_NAMES = [
    # Frequency (F0 / pitch)
    "F0 Mean (Hz)", "F0 Std Dev", "F0 Percentile 20", "F0 Percentile 50",
    "F0 Percentile 80", "F0 Range", "F0 Mean Rising Slope",
    "F0 Mean Falling Slope",
    # Energy / Loudness
    "Loudness Mean", "Loudness Std Dev", "Loudness Percentile 20",
    "Loudness Percentile 50", "Loudness Percentile 80",
    "Loudness Range", "Loudness Rising Slope", "Loudness Falling Slope",
    # Spectral balance
    "Spectral Flux Mean", "Spectral Flux Std",
    "MFCC1 Mean", "MFCC1 Std", "MFCC2 Mean", "MFCC2 Std",
    "MFCC3 Mean", "MFCC3 Std", "MFCC4 Mean", "MFCC4 Std",
    # Cepstral (voice quality / timbre)
    "MFCC5 Mean", "MFCC5 Std", "MFCC6 Mean", "MFCC6 Std",
    "MFCC7 Mean", "MFCC7 Std", "MFCC8 Mean", "MFCC8 Std",
    "MFCC9 Mean", "MFCC9 Std", "MFCC10 Mean", "MFCC10 Std",
    "MFCC11 Mean", "MFCC11 Std", "MFCC12 Mean", "MFCC12 Std",
    # Voice quality
    "Jitter (local)", "Jitter (DDP)", "Shimmer (local)",
    "Shimmer (dB)", "HNR (dB) Mean", "HNR (dB) Std",
    # Formants
    "F1 Freq Mean", "F1 Freq Std", "F1 Bandwidth Mean", "F1 Bandwidth Std",
    "F1 Amplitude Mean", "F1 Amplitude Std",
    "F2 Freq Mean", "F2 Freq Std", "F2 Bandwidth Mean", "F2 Bandwidth Std",
    "F2 Amplitude Mean", "F2 Amplitude Std",
    "F3 Freq Mean", "F3 Freq Std", "F3 Bandwidth Mean", "F3 Bandwidth Std",
    "F3 Amplitude Mean", "F3 Amplitude Std",
    # Alpha ratio / Hammarberg index
    "Alpha Ratio Mean", "Alpha Ratio Std",
    "Hammarberg Index Mean", "Hammarberg Index Std",
    "Spectral Slope 0-500 Mean", "Spectral Slope 0-500 Std",
    "Spectral Slope 500-1500 Mean", "Spectral Slope 500-1500 Std",
    # Zero crossing / temporal
    "ZCR Mean", "ZCR Std",
    "Voiced Segments Mean", "Unvoiced Segments Mean",
    "Voiced Segments Std", "Unvoiced Segments Std",
    "Voiced Rate", "Pause Rate",
    # RMS / energy
    "RMS Energy Mean", "RMS Energy Std",
    "Energy Slope Mean", "Energy Slope Std",
    "Log Energy Mean", "Log Energy Std",
    "Spectral Centroid Mean", "Spectral Centroid Std",
]


def extract_egemaps_features(audio_path: str) -> np.ndarray:
    """
    Extract 88 eGeMAPS features using opensmile.
    Returns shape (1, 88) — one aggregated vector per file.
    """
    try:
        import opensmile
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,  # aggregated over file
        )
        features_df = smile.process_file(audio_path)
        features = features_df.values.astype(np.float32)   # (1, 88)
        return np.nan_to_num(features)
    except ImportError:
        print("opensmile not installed — falling back to librosa")
        return None
    except Exception as e:
        print(f"opensmile error: {e} — falling back to librosa")
        return None


def extract_librosa_fallback(audio_path: str, sr: int = 16000) -> np.ndarray:
    """
    Fallback feature extraction using librosa.
    Matches the original covarep-like 39-feature set.
    """
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=sr)
        features_list = []
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features_list.append(mfcc.T)
        features_list.append(librosa.feature.delta(mfcc).T)
        features_list.append(librosa.feature.spectral_centroid(y=y, sr=sr).T)
        features_list.append(librosa.feature.spectral_bandwidth(y=y, sr=sr).T)
        features_list.append(librosa.feature.spectral_rolloff(y=y, sr=sr).T)
        features_list.append(librosa.feature.spectral_contrast(y=y, sr=sr).T)
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr, frame_length=2048)
        features_list.append(np.nan_to_num(f0).reshape(-1, 1))
        features_list.append(librosa.feature.rms(y=y).T)
        features_list.append(librosa.feature.zero_crossing_rate(y).T)
        min_len  = min(f.shape[0] for f in features_list)
        features = np.hstack([f[:min_len] for f in features_list])
        return np.nan_to_num(features).astype(np.float32)
    except Exception as e:
        print(f"Librosa fallback error: {e}")
        return None


def extract_raw_features_for_shap(audio_path: str, audio_config: dict) -> np.ndarray:
    """
    Extract aggregated raw (unscaled) features for SHAP explainability.
    Returns (1, n_features) — one vector per recording.

    Uses eGeMAPS if available, falls back to librosa.
    """
    # Try eGeMAPS first
    features = extract_egemaps_features(audio_path)
    if features is not None:
        # eGeMAPS already returns (1, 88) functionals
        return features

    # Fallback: librosa → aggregate by mean
    features = extract_librosa_fallback(audio_path)
    if features is None:
        return None

    expected_dim = audio_config.get("input_dim", features.shape[1])
    if features.shape[1] != expected_dim:
        if features.shape[1] < expected_dim:
            pad      = np.zeros((features.shape[0], expected_dim - features.shape[1]))
            features = np.hstack([features, pad])
        else:
            features = features[:, :expected_dim]

    return features.mean(axis=0, keepdims=True).astype(np.float32)


def process_audio_for_model(audio_path, audio_scaler, audio_model, audio_config):
    """
    Full audio processing pipeline:
    1. Extract features (eGeMAPS preferred, librosa fallback)
    2. Pad/truncate to model's expected shape
    3. Scale and run inference
    Returns (embedding, confidence)
    """
    # Try eGeMAPS
    egemaps = extract_egemaps_features(audio_path)
    if egemaps is not None:
        # eGeMAPS → (1, 88). Model expects (batch, seq_len, input_dim)
        features_frame = egemaps  # (1, 88)
        expected_dim   = audio_config["input_dim"]
        n_feat         = features_frame.shape[1]
        # Always align to model's expected input dimension
        if n_feat > expected_dim:
            features_frame = features_frame[:, :expected_dim]
        elif n_feat < expected_dim:
            pad            = np.zeros((1, expected_dim - n_feat))
            features_frame = np.hstack([features_frame, pad])
        # Expand to seq_len=512 (repeat single frame — model needs time dimension)
        features_seq = np.repeat(features_frame[:, np.newaxis, :], 512, axis=1)
    else:
        # Librosa fallback — original pipeline
        features = extract_librosa_fallback(audio_path)
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
        if len(features) > 512:
            features = features[:512]
        else:
            pad      = np.zeros((512 - len(features), features.shape[1]))
            features = np.vstack([features, pad])
        features_seq = features[np.newaxis]  # (1, 512, dim)

    # Scale
    try:
        seq_len, dim = features_seq.shape[1], features_seq.shape[2]
        flat    = features_seq.reshape(-1, dim)
        scaled  = audio_scaler.transform(flat)
        features_seq = scaled.reshape(1, seq_len, dim)
    except Exception:
        pass

    tensor = torch.tensor(features_seq, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        embedding  = audio_model.get_embeddings(tensor)
        logits     = audio_model(tensor)
        probs      = torch.softmax(logits, dim=1)
        confidence = probs[0][1].item()

    return embedding.cpu().numpy(), confidence


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe audio using Whisper (offline, accurate).
    Falls back to SpeechRecognition (Google API) if whisper not installed.
    """
    # Try Whisper first (offline, better accuracy)
    try:
        import warnings, logging
        warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")
        logging.getLogger("whisper").setLevel(logging.ERROR)
        import whisper
        model = whisper.load_model("base")   # ~150MB, loads once
        result = model.transcribe(audio_path, language="en", fp16=False)
        return result["text"].strip()
    except ImportError:
        pass
    except Exception as e:
        print(f"Whisper error: {e}")

    # Fallback: SpeechRecognition (requires internet)
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)
    except Exception as e:
        print(f"Transcription fallback error: {e}")
        return ""
