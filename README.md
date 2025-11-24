# 🎭 Real-Time Gender Detection System

A high-performance gender detection system combining **YOLO face detection**, **ArcFace embeddings**, and **dual classification models** (FAISS + MLP Neural Network) for accurate real-time predictions.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

📄 **[Read the Conference Paper](https://www.researchgate.net/publication/397702586_Real-Time_Gender_Classification_Using_Deep_Metric_Learning_and_Vector_Database_Retrieval_A_Hybrid_Ensemble_Approach?_sg=NcRFw10sNzA5ewh1l3yTm0ZdrPuFgS_4rEIaLYf7D8lUkhcD8_WsJ54obEVaVoTS7SJKSl1lXiap3K26bCd08klPUZ2Pquytf5cloNgH.9Z5CVyoaYdcm59x_DyB0bSfHeCuAY-TNfbYyKBTnZwXqoMAgNj1P2RZdOv5jscO2bKskBOS2RYxzIMEr-MkyRQ&_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6ImhvbWUiLCJwYWdlIjoiX2RpcmVjdCJ9fQ)** | *Real-Time Gender Classification Using Deep Metric Learning and Vector Database Retrieval: A Hybrid Ensemble Approach*

---

## 🎬 Demo Video

🎥 **[Watch the Live Demo](https://drive.google.com/file/d/1VI4dvORVyvUvj8AMzr9VBfGbbX7XR_Vn/view?usp=sharing)** - See the system in action with real-time gender classification!

---

## ✨ Features

- 🚀 **Real-time Detection**: Process live webcam feed with minimal latency
- 🎯 **High Accuracy**: Dual-model ensemble approach (FAISS k-NN + MLP Neural Network)
- 🔍 **Advanced Face Detection**: YOLOv11 for robust face detection
- 🧠 **Deep Embeddings**: ArcFace (buffalo_l) for powerful facial feature extraction
- ⚙️ **Configurable**: Easy-to-customize YAML configuration
- 📊 **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and ROC curves
- 🎨 **Visual Feedback**: Color-coded bounding boxes with confidence scores

---

## 🏗️ Architecture

```
┌─────────────────┐
│  Video Input    │
│  (Webcam/File)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   YOLOv11 Face  │
│    Detection    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    ArcFace      │
│   Embedding     │
│   (512-dim)     │
└────────┬────────┘
         │
    ┌────┴─────┐
    ▼          ▼
┌───────┐  ┌───────┐
│ FAISS │  │  MLP  │
│  k-NN │  │ Model │
└───┬───┘  └───┬───┘
    │          │
    └────┬─────┘
         ▼
    ┌─────────┐
    │Ensemble │
    │  Vote   │
    └────┬────┘
         │
         ▼
    🎭 Gender
```

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster processing)
- Webcam (for real-time detection)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/gender-detection.git
   cd gender-detection
   ```

2. **Install dependencies**
   ```bash
   pip install torch torchvision torchaudio
   pip install ultralytics opencv-python numpy pandas
   pip install faiss-cpu  # or faiss-gpu for CUDA support
   pip install onnxruntime  # or onnxruntime-gpu
   pip install pyyaml scikit-learn matplotlib seaborn
   ```

3. **Download Pre-trained Models**
   
   Download the required models from [GitHub Releases](https://github.com/menuka400/Real-Time-Gender-Classification-Using-Deep-Metric-Learning-and-Vector-Database-Retrieval/releases):
   
   - `yolov11l-face.pt` → Place in `Pre-Train Models/`
   - `buffalo_l.zip` → Extract to `Pre-Train Models/models/buffalo_l/`
   - `face_embeddings.index` → Place in `embeddings/` folder
   
   ```bash
   # After downloading, organize the files:
   mkdir -p "Pre-Train Models/models/buffalo_l"
   mkdir -p embeddings
   
   # Extract buffalo_l.zip to the correct location
   # Move face_embeddings.index to embeddings folder
   ```

---

## 🚀 Quick Start

### Real-Time Gender Detection

Run the detection system on your webcam:

```bash
python realtime_gender_detection.py
```

**Controls:**
- Press `Q` to quit
- Adjust settings in `config.yaml`

### Train MLP Model

Train the MLP neural network on your embeddings:

```bash
python train_mlp_model.py
```

This will:
- Load face embeddings from FAISS index
- Train a Multi-Layer Perceptron classifier
- Save the model to `models/mlp_gender_model.pth`
- Generate training metrics and visualizations

### Evaluate Models

Evaluate both FAISS and MLP models:

```bash
python evaluate_gender_model.py
```

Results saved in `evaluation_results/`:
- Accuracy, Precision, Recall, F1-Score
- Confusion matrices
- ROC curves and AUC scores
- Model comparison report

---

## ⚙️ Configuration

Edit `config.yaml` to customize system behavior:

```yaml
face_detection:
  confidence_threshold: 0.8  # Detection sensitivity

gender_classification:
  male_threshold: 0.7        # Male classification threshold
  female_threshold: 0.7      # Female classification threshold
  k_neighbors: 5             # k-NN neighbors
  use_ensemble: true         # Enable ensemble mode
  faiss_weight: 0.5          # FAISS model weight
  mlp_weight: 0.5            # MLP model weight

display:
  show_confidence: true      # Show confidence scores
  show_detailed_info: true   # Show model breakdown
```

---

## 📊 Model Performance

### Evaluation Metrics

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| FAISS k-NN | 94.2% | 93.8% | 94.5% | 94.1% |
| MLP Neural Net | 95.7% | 95.3% | 96.0% | 95.6% |
| **Ensemble** | **96.5%** | **96.2%** | **96.8%** | **96.5%** |

*Results may vary based on dataset and configuration*

---

## 📁 Project Structure

```
AGE/
├── config.yaml                      # System configuration
├── realtime_gender_detection.py    # Main detection script
├── train_mlp_model.py               # MLP training script
├── evaluate_gender_model.py         # Model evaluation script
├── embeddings/
│   ├── face_embeddings.index        # FAISS index
│   └── checkpoint.json              # Training metadata
├── models/
│   └── mlp_gender_model.pth         # Trained MLP model
├── Pre-Train Models/
│   ├── yolov11l-face.pt             # YOLO face detector
│   └── models/buffalo_l/            # ArcFace models
│       └── w600k_r50.onnx
└── evaluation_results/
    ├── faiss_metrics_report.txt
    ├── mlp_metrics_report.txt
    └── model_comparison.csv
```

---

## 🔧 Technical Details

### Models Used

1. **YOLOv11-Face**: State-of-the-art face detection
2. **ArcFace (buffalo_l)**: 512-dimensional facial embeddings
3. **FAISS k-NN**: Fast similarity search for gender classification
4. **MLP Classifier**: Deep neural network (512→256→128→64→2)

### Key Technologies

- **PyTorch**: Neural network training and inference
- **FAISS**: Efficient similarity search
- **ONNX Runtime**: Optimized model inference
- **OpenCV**: Computer vision operations
- **Ultralytics**: YOLO implementation

---

## 📈 Use Cases

- 👥 **Demographic Analysis**: Analyze customer demographics in retail
- 🔐 **Access Control**: Gender-based access systems
- 📊 **Analytics**: Real-time audience analysis at events
- 🎮 **Gaming**: Interactive gaming experiences
- 🔬 **Research**: Gender bias studies and dataset analysis

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **YOLO**: Ultralytics team for YOLOv11
- **ArcFace**: InsightFace for buffalo_l models
- **FAISS**: Facebook AI Research for efficient similarity search
- **PyTorch**: PyTorch team for the deep learning framework

---

## 📄 Citation

If you use this project in your research, please cite our conference paper:

```bibtex
@inproceedings{realtime_gender_classification_2025,
  title={Real-Time Gender Classification Using Deep Metric Learning and Vector Database Retrieval: A Hybrid Ensemble Approach},
  author={[Menuka Hashen Jayasuriya]},
  booktitle={[Proceedings of the Independent Study on AI and Computer Vision]},
  year={2025},
  url={https://www.researchgate.net/publication/397676791}
}
```

**Paper**: [Real-Time Gender Classification Using Deep Metric Learning and Vector Database Retrieval](https://www.researchgate.net/publication/397702586_Real-Time_Gender_Classification_Using_Deep_Metric_Learning_and_Vector_Database_Retrieval_A_Hybrid_Ensemble_Approach?_sg=NcRFw10sNzA5ewh1l3yTm0ZdrPuFgS_4rEIaLYf7D8lUkhcD8_WsJ54obEVaVoTS7SJKSl1lXiap3K26bCd08klPUZ2Pquytf5cloNgH.9Z5CVyoaYdcm59x_DyB0bSfHeCuAY-TNfbYyKBTnZwXqoMAgNj1P2RZdOv5jscO2bKskBOS2RYxzIMEr-MkyRQ&_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6ImhvbWUiLCJwYWdlIjoiX2RpcmVjdCJ9fQ)

---

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

<div align="center">
  <strong>⭐ Star this repository if you find it helpful! ⭐</strong>
</div>
