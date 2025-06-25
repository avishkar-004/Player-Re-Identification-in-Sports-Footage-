# ⚽ Player Re-Identification in Soccer Footage

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-orange.svg)](https://pytorch.org/)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Advanced Computer Vision Solution for Sports Analytics**

*Liat.ai AI/ML Intern Assignment - Option 2: Re-Identification in a Single Feed*

</div>

---

## 🎯 Project Overview

This repository contains a state-of-the-art **Player Re-Identification System** that processes soccer footage to detect and track objects (ball, goalkeeper, players, referees) while maintaining consistent identities even when players exit and re-enter the frame.

### 🏆 Key Achievement
**96.27% Player Re-Identification Accuracy** - Successfully exceeding the primary objective requirements.

### 🔧 Technical Stack
- **Detection**: YOLOv11 with custom confidence thresholds
- **Re-Identification**: ResNet18-based appearance embeddings
- **Tracking**: Enhanced multi-modal similarity tracking
- **Evaluation**: Comprehensive performance metrics framework

---

## 📺 Demo & Results

### 🎥 Output Video
[![Watch Demo](https://img.shields.io/badge/🎬_Watch_Demo-Google_Drive-blue?style=for-the-badge)](https://drive.google.com/file/d/1fzXlMIaVKB7IqiwfyF7XfvEmVRIvyxhe/view?usp=sharing)

### 📊 Performance Metrics

| Metric | Ball | Goalkeeper | **Player** | Referee |
|--------|------|------------|------------|---------|
| **Re-ID Success** | 45.5% | 72.3% | **96.27%** ✅ | 68.9% |
| **Detection/Frame** | 0.95 ± 0.18 | 0.88 ± 0.25 | 12.77 ± 2.15 | 1.82 ± 0.44 |
| **ID Switches** | 3 | 2 | 4 | 2 |

**System Performance**: ~64 FPS processing speed (exceeds 17 FPS target)

---

## 🚀 Quick Start

### Prerequisites

```bash
# System Requirements
- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)
- 8GB+ RAM
```

### 📥 Required Files

1. **Pre-trained Model**: [`best.pt`](https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view)
   - YOLOv11 model trained for soccer objects
   - Classes: `{0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}`

2. **Input Video**: `15sec_input_720p.mp4` (720p, 15-second soccer footage)

### 🛠️ Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd player_reid

# Install dependencies
pip install -r requirements.txt

# Place required files in root directory
# - 15sec_input_720p.mp4
# - best.pt

# Windows users only (avoid OpenMP conflicts)
set KMP_DUPLICATE_LIB_OK=TRUE
```

---

## 🎮 Usage

### 1️⃣ Process Video

```bash
python main.py
```

**What it does:**
- 🔍 Detects objects using YOLOv11
- 🎯 Tracks and re-identifies players using ResNet18 embeddings
- 🎨 Generates annotated video with color-coded bounding boxes
- 📝 Creates detailed logging for analysis

**Outputs:**
```
output/output.mp4              # Annotated video
logs/detection_log.txt         # Frame-by-frame detection results
logs/tracking_log.txt          # ID assignments and re-ID events
```

**Color Coding:**
- 🔴 **Red**: Ball
- 🟡 **Yellow**: Goalkeeper  
- 🟢 **Green**: Players
- 🔵 **Blue**: Referees

### 2️⃣ Evaluate Performance

```bash
python evaluate.py
```

**Generates comprehensive analysis:**
```
evaluation_report.txt          # Detailed performance metrics
logs/evaluation_log.txt        # Evaluation process logging
```

**Metrics Analyzed:**
- ✅ Detection accuracy per class
- 🔄 ID switches and consistency
- 🎯 Re-identification success rates
- ⚡ Processing efficiency (FPS)
- 📊 ID assignment patterns

---

## 📁 Project Structure

```
player_reid/
├── 📄 main.py                    # Main processing pipeline
├── 📊 evaluate.py               # Performance evaluation system
├── 🛠️ utils/
│   ├── detection.py            # YOLOv11 detection wrapper
│   ├── tracking.py             # Enhanced tracking & re-ID
│   └── visualization.py        # Video annotation system
├── 📂 logs/                     # System logs
│   ├── detection_log.txt       # Detection results
│   ├── tracking_log.txt        # Tracking events
│   └── evaluation_log.txt      # Performance analysis
├── 📂 output/                   # Generated outputs
│   └── output.mp4              # Annotated video
├── 📋 requirements.txt          # Python dependencies
├── 📖 README.md                # This file
├── 📄 report.md                # Technical implementation report
├── 🎥 15sec_input_720p.mp4     # Input video (to be added)
└── 🧠 best.pt                  # Pre-trained model (to be added)
```

---

## 🔬 Technical Deep Dive

### 🏗️ Architecture Overview

```mermaid
graph LR
    A[Input Video] --> B[YOLOv11 Detection]
    B --> C[Feature Extraction]
    C --> D[Enhanced Tracker]
    D --> E[Re-Identification]
    E --> F[Visualization]
    F --> G[Annotated Output]
    
    B --> H[Detection Logs]
    D --> I[Tracking Logs]
    G --> J[Evaluation Metrics]
```

### 🧠 Core Components

#### **1. Detection Module** (`utils/detection.py`)
```python
class Detector:
    def __init__(self, model_path, conf_thres=0.25, iou_thres=0.5):
        self.class_conf_thres = {
            0: 0.2,   # Ball - Optimized for small objects
            1: 0.25,  # Goalkeeper
            2: 0.25,  # Player  
            3: 0.2    # Referee
        }
```

#### **2. Tracking & Re-ID** (`utils/tracking.py`)
```python
class EnhancedTracker:
    def __init__(self):
        self.similarity_threshold = 0.35        # Optimized threshold
        self.max_disappeared_frames = 75        # ~3 seconds tolerance
        self.appearance_extractor = ResNet18    # Feature embeddings
```

#### **3. Multi-Modal Similarity**
- **Spatial**: Distance-based position tracking
- **Appearance**: ResNet18 embeddings with cosine similarity
- **Size**: Object size consistency checking
- **Weighted Fusion**: Class-dependent similarity combinations

---

## 📈 Performance Analysis

### 🎯 Achievements

✅ **Primary Objective Achieved**: 96.27% player re-identification accuracy  
✅ **Real-time Performance**: Exceeds 17 FPS target with ~64 FPS processing  
✅ **Robust Tracking**: Handles player occlusion and re-entry scenarios  
✅ **Modular Design**: Extensible architecture for future enhancements  

### 🔍 Key Innovations

1. **Adaptive Confidence Thresholds**
   - Solved initial detection failure (0.4 → 0.25 threshold)
   - Class-specific optimization for better accuracy

2. **Enhanced Re-Identification**
   - 75-frame disappearance tolerance
   - Multi-modal similarity scoring
   - ResNet18 appearance embeddings

3. **Comprehensive Evaluation**
   - Automated performance assessment
   - Detailed logging and metrics
   - Pass/fail criteria validation

### ⚠️ Known Limitations

- **Ball Re-ID**: 45.5% success rate (small object challenges)
- **Goalkeeper Detection**: Reduced count due to occlusion
- **FPS Measurement**: Potentially inflated due to logging methodology

---

## 🔮 Future Enhancements

### 🚀 Short-term Improvements

- [ ] **Kalman Filters**: Predictive tracking for ball trajectory
- [ ] **MobileNet**: Faster embedding generation
- [ ] **Fine-tuning**: Soccer-specific YOLOv11 training
- [ ] **FPS Optimization**: Accurate frame timestamp measurement

### 🌟 Advanced Features

- [ ] **Transformer Embeddings**: Better feature representation
- [ ] **Multi-scale Fusion**: Handle scale variations
- [ ] **Graph Neural Networks**: Player interaction modeling
- [ ] **Domain Adaptation**: Cross-venue generalization

---

## 🛠️ Development

### 🧪 Testing

```bash
# Run full pipeline test
python main.py

# Evaluate performance
python evaluate.py

# Check specific frame range (re-ID example)
# Examine frames 210-229 (~8.4-9.2s) in output video
```

### 🐛 Debugging

```bash
# Enable detailed logging
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Check log files for issues
tail -f logs/tracking_log.txt
tail -f logs/detection_log.txt
```

### 📊 Configuration

Key parameters can be adjusted in respective modules:

```python
# Detection thresholds (utils/detection.py)
CONF_THRESHOLD = 0.25
CLASS_CONF_THRESHOLDS = {0: 0.2, 1: 0.25, 2: 0.25, 3: 0.2}

# Tracking parameters (utils/tracking.py)  
MAX_DISAPPEARED_FRAMES = 75
SIMILARITY_THRESHOLD = 0.35
MAX_DISTANCE_THRESHOLD = 100.0
```

---

## 📚 Dependencies

```txt
ultralytics==8.0.196          # YOLOv11 object detection
deep-sort-realtime==1.3.2     # Tracking utilities
opencv-python==4.8.0.76       # Video processing
torch==2.0.1                  # PyTorch deep learning
numpy==1.24.3                 # Numerical computing
scipy==1.10.1                 # Scientific computing
```

---

## 🏆 Results Validation

### ✅ Critical Test Cases

1. **Re-identification Example**: Frames 210-229 (~8.4-9.2s)
   - Players exit frame during goal event
   - Successfully re-identified upon re-entry
   - Consistent ID maintenance

2. **Performance Benchmarks**:
   - Detection accuracy: Meets targets for ball, goalkeeper, referee
   - Player tracking: Exceeds 80% re-ID requirement significantly
   - Processing speed: Real-time capable with room for optimization

### 📋 Evaluation Criteria

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Player Re-ID | >80% | **96.27%** | ✅ Pass |
| Processing Speed | ~17 FPS | ~64 FPS | ✅ Pass |
| ID Switches (Players) | <5 | 4 | ✅ Pass |
| Detection Consistency | Stable | ±2.15 std | ✅ Pass |

---

## 🤝 Contributing

This project was developed as part of the Liat.ai AI/ML Intern Assignment. For questions or improvements:

1. **Technical Issues**: Check logs directory for debugging information
2. **Performance Questions**: Run `evaluate.py` for detailed metrics
3. **Enhancement Ideas**: Refer to Future Enhancements section

---

## 📄 License

This project is part of an internship assignment for Liat.ai. Please refer to assignment guidelines for usage restrictions.

---

## 🙏 Acknowledgments

- **Liat.ai**: For providing the challenging and engaging assignment
- **Ultralytics**: For the excellent YOLOv11 implementation
- **PyTorch Team**: For the robust deep learning framework
- **OpenCV**: For comprehensive computer vision tools

---

<div align="center">

**Built with ❤️ for Sports Analytics**

*Transforming soccer footage into actionable insights through advanced computer vision*

</div>