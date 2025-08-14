# 🚀 Quick Setup Guide - Cow Face ID Model

## 📋 Prerequisites

- **Python 3.8 or higher** installed
- **Webcam or USB camera** for real-time recognition
- **4GB RAM minimum** (8GB recommended)
- **Windows, Linux, or macOS**

## ⚡ 5-Minute Setup

### 1. Clone Repository
```bash
git clone https://github.com/gautamk1512/Cow-Face-Id-End-to-end-Model.git
cd Cow-Face-Id-End-to-end-Model
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Test Your Setup
```bash
# Test camera connectivity
python test_camera.py

# Test basic detection
python simple_cow_camera.py --camera 0

# Test smart talking camera
python smart_talking_camera.py --camera 0 --threshold 0.7
```

## 🎯 Quick Start Commands

### For Beginners
```bash
# Simple demo (no model required)
python simple_cow_camera.py --camera 0
```

### For AI Enthusiasts  
```bash
# Smart talking camera with voice
python smart_talking_camera.py --camera 0
```

### For Researchers
```bash
# Full recognition system (requires trained model)
python camera_cow_recognition.py --checkpoint runs/checkpoints/best.pt --gallery data/val --camera 0
```

## 🔧 Troubleshooting

### Camera Issues
```bash
# Try different camera IDs
python test_camera.py 0  # Built-in webcam
python test_camera.py 1  # External USB camera
```

### Installation Issues
```bash
# If torch installation fails
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# If OpenCV issues
pip uninstall opencv-python
pip install opencv-python-headless
```

### Voice Issues (Windows)
```bash
# Install additional TTS components
pip install pypiwin32
```

## 📱 Usage Examples

### Demo Mode (No Training Required)
```bash
python smart_talking_camera.py --camera 0
# - Shows face detection
# - Simulates cow recognition
# - Provides voice announcements
```

### Training Mode (Custom Dataset)
```bash
# 1. Organize your data in data/train/ and data/val/
# 2. Train the model
python -m src.training.train_classifier --data_root data --config configs/default.yaml --epochs 20

# 3. Test with your trained model
python camera_cow_recognition.py --checkpoint runs/checkpoints/best.pt --gallery data/val --camera 0
```

## ✅ Verification Steps

After setup, verify everything works:

1. **Camera Test**: `python test_camera.py` - Should show video feed
2. **Detection Test**: `python simple_cow_camera.py` - Should show detection boxes
3. **Voice Test**: `python smart_talking_camera.py` - Should hear announcements
4. **Model Test**: `python demo_cow_face_id.py` - Should show recognition results

## 🆘 Need Help?

- **Documentation**: Check `README.md` for detailed information
- **Camera Guide**: See `CAMERA_USAGE_GUIDE.md` for camera-specific help
- **Issues**: [Report bugs on GitHub](https://github.com/gautamk1512/Cow-Face-Id-End-to-end-Model/issues)

## 🎉 You're Ready!

Once setup is complete, you can:
- ✅ Detect faces in real-time
- ✅ Hear voice announcements
- ✅ Train custom cow recognition models
- ✅ Use multiple camera applications
- ✅ Deploy for farm monitoring or research

**Start with the smart talking camera for the best demo experience:**
```bash
python smart_talking_camera.py --camera 0 --threshold 0.7
```

Happy cow recognition! 🐄✨
