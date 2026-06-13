import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# --------------------------------------------------------
st.set_page_config(
    page_title="Deepfake Audio Detector",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ Deepfake Audio Detector")
st.markdown("""
Detect whether an audio recording is **Genuine (Human)** or **Deepfake (AI-Generated)**. 
Upload a `.wav` file below to analyze its acoustic footprint.
""")

# --------------------------------------------------------
# 2. CACHED MODEL LOADING
# --------------------------------------------------------
@st.cache_resource
def load_deepfake_model():
    # Loads the trained Keras model saved from the Kaggle notebook
    try:
        model = tf.keras.models.load_model('deepfake_audio_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model file 'deepfake_audio_model.keras': {e}")
        return None

model = load_deepfake_model()

# --------------------------------------------------------
# 3. FEATURE EXTRACTION PIPELINE
# --------------------------------------------------------
def preprocess_audio(file_path, max_pad_len=400):
    # Must perfectly mirror the preprocessing steps used during training
    audio, sample_rate = librosa.load(file_path, sr=16000)
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    
    # Pad or truncate to ensure uniform shape
    if mfccs.shape[1] > max_pad_len:
        mfccs = mfccs[:, :max_pad_len]
    else:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
    # Reshape and transpose to match Conv1D expected input format: (1, max_pad_len, n_mfcc)
    mfccs = np.expand_dims(mfccs, axis=0)
    mfccs = np.transpose(mfccs, (0, 2, 1))
    return audio, sample_rate, mfccs

# --------------------------------------------------------
# 4. USER INTERFACE & INTERACTION
# --------------------------------------------------------
uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

if uploaded_file is not None and model is not None:
    st.info("Processing file and generating prediction...")
    
    # Preprocess the uploaded audio file
    try:
        audio, sr, processed_features = preprocess_audio(uploaded_file)
        
        # Run prediction
        prediction_prob = model.predict(processed_features)[0][0]
        
        # --------------------------------------------------------
        # 5. DISPLAY RESULTS
        # --------------------------------------------------------
        st.subheader("Analysis Results")
        
        # Calculate confidence metric
        # Model outputs close to 1 mean Deepfake, close to 0 mean Genuine
        if prediction_prob >= 0.5:
            label = "🔴 Deepfake (AI-Generated)"
            confidence = prediction_prob * 100
        else:
            label = "🟢 Genuine (Human)"
            confidence = (1 - prediction_prob) * 100
            
        # UI Cards for Results
        st.metric(label="Classification Decision", value=label)
        st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
        
        # Audio Playback
        st.write("### Audio Playback")
        st.audio(uploaded_file, format='audio/wav')
        
        # Waveform Visualization
        st.write("### Acoustic Waveform")
        fig, ax = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(audio, sr=sr, ax=ax)
        ax.set_title("Audio Signal Waveform")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"An error occurred during verification: {e}")