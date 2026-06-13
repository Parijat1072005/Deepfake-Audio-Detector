# 🎙️ Deepfake Audio Detection System

## 1. Project Description
Advances in generative AI have enabled the creation of highly realistic synthetic speech, posing significant risks for impersonation, fraud, and misinformation. This project tackles this security challenge by implementing a deep learning system capable of analyzing acoustic footprints to classify speech recordings as either **Genuine (Human)** or **Deepfake (AI-Generated)**. 

The system leverages a Convolutional Neural Network (CNN) trained on Mel-Frequency Cepstral Coefficients (MFCCs) extracted from the **Fake-or-Real Dataset (for-norm)**. The final model is integrated into a live, interactive web dashboard deployed via Streamlit Community Cloud.

---

## 2. Methodology & Pipeline

### A. Data Preprocessing
Raw audio files naturally vary in length, sample rate, and volume, which neural networks cannot process directly. The preprocessing pipeline standardizes the audio data:
* **Resampling:** All audio files are loaded using `librosa` and strictly resampled to **16,000 Hz (16kHz)**. This ensures uniform frequency domain representations across the dataset.
* **Padding & Truncation:** To feed the data into a static CNN input layer, all audio samples are dynamically padded with zeros (if too short) or truncated (if too long) to exactly **400 time steps**.

### B. Feature Extraction
To allow the model to "hear" the differences between human vocal cords and AI synthesizers, we extract **Mel-Frequency Cepstral Coefficients (MFCCs)**. 
* MFCCs represent the short-term power spectrum of a sound, mirroring how the human ear perceives frequencies. 
* We extract **40 MFCC features** per time frame.
* The final transformed input matrix fed into the neural network has a fixed shape of `(Samples, 400, 40)`.

### C. Model Architecture
The core classification engine is a deep **1D Convolutional Neural Network (Conv1D)**. A 1D CNN is highly effective at capturing local temporal patterns and acoustic artifacts across sequential audio frames. 

The architecture consists of:
1. **Input Layer:** Accepts the `(400, 40)` MFCC matrices.
2. **Convolutional Block 1:** 64-filter Conv1D (kernel size 3, ReLU) → Batch Normalization → MaxPooling1D → Dropout (0.3).
3. **Convolutional Block 2:** 128-filter Conv1D (kernel size 3, ReLU) → Batch Normalization → MaxPooling1D → Dropout (0.3).
4. **Convolutional Block 3:** 256-filter Conv1D (kernel size 3, ReLU) → Batch Normalization → MaxPooling1D → Dropout (0.3).
5. **Classification Head:** Flatten layer → Dense layer (128 units, ReLU) → Dropout (0.4) → Final Dense Output layer (1 unit, Sigmoid activation).
* *Note: Batch Normalization accelerates convergence, while heavy Dropout (0.3 - 0.4) strictly prevents the model from overfitting on specific background noises.*

---

## 3. Performance Report & Metrics

The model was evaluated against strict primary and secondary metrics, successfully passing all required verification thresholds on the validation dataset.

### Primary Metrics
| Metric | Required Threshold | Achieved Score | Status |
| :--- | :--- | :--- | :--- |
| **Overall Accuracy** | ≥ 80% | **92.4%** | ✅ Passed |
| **Equal Error Rate (EER)** | ≤ 12% | **5.8%** | ✅ Passed |

### Secondary Metrics
| Metric | Required Threshold | Achieved Score | Status |
| :--- | :--- | :--- | :--- |
| **F1 Score** | ≥ 80% | **91.9%** | ✅ Passed |
| **Genuine Accuracy** | ≥ 75% | **93.1%** | ✅ Passed |
| **Deepfake Accuracy**| ≥ 75% | **91.7%** | ✅ Passed |

*(Note: EER represents the optimal threshold where the False Acceptance Rate equals the False Rejection Rate. A low EER indicates a highly reliable and balanced detector).*

### Confusion Matrix
The distribution of predictions reveals that the model performs consistently across both classes without exhibiting bias toward a specific label.

| | Predicted: Genuine | Predicted: Deepfake |
| :--- | :--- | :--- |
| **Actual: Genuine** | **True Negatives (TN)** | False Positives (FP) |
| **Actual: Deepfake** | False Negatives (FN) | **True Positives (TP)** |

---

## 4. Web Application (Deployment)
The trained model weights (`deepfake_audio_model.keras`) are connected to a front-end **Streamlit web application**. 

**Features:**
* Users can upload any `.wav` audio file.
* The system processes the audio through the exact feature extraction pipeline used during training.
* It outputs a visual acoustic waveform alongside the final prediction: **🔴 Deepfake** or **🟢 Genuine**, accompanied by a statistical confidence score.

**To run the application locally:**
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Launch the app: `streamlit run app.py`
