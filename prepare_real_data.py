#!/usr/bin/env python3
"""
Real Data Preparation Script
Helps organize real human and cow face images for perfect model training
"""

import os
import sys
import cv2
import shutil
from pathlib import Path
import argparse
from tqdm import tqdm
import json

# Add src to path
sys.path.append('src')

from src.detection.detect_faces import YOLOv5Detector

class RealDataPreparator:
    def __init__(self):
        self.detector = YOLOv5Detector()
        self.face_count = 0
        
        print("🎯 REAL DATA PREPARATION SYSTEM")
        print("=" * 50)
        print("📸 Preparing real human and cow faces for training")
        print("🎯 Goal: Perfect distinction between humans and cows")
        print()

    def create_data_structure(self):
        """Create proper data directory structure"""
        directories = [
            "data/raw/humans",
            "data/raw/cows", 
            "data/train/human",
            "data/train/cow_001",
            "data/train/cow_002", 
            "data/train/cow_003",
            "data/val/human",
            "data/val/cow_001",
            "data/val/cow_002",
            "data/val/cow_003"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
        print("📁 Created data directory structure:")
        for directory in directories:
            print(f"   ✓ {directory}")
        
        # Create data organization guide
        guide_content = """# Real Data Organization Guide

## 📁 Directory Structure

### Raw Data Collection
- `data/raw/humans/` - Put all raw human photos here
- `data/raw/cows/` - Put all raw cow photos here

### Processed Training Data  
- `data/train/human/` - Human face crops for training
- `data/train/cow_001/` - Cow 1 face crops for training
- `data/train/cow_002/` - Cow 2 face crops for training
- `data/train/cow_003/` - Cow 3 face crops for training

### Validation Data
- `data/val/human/` - Human face crops for validation  
- `data/val/cow_001/` - Cow 1 face crops for validation
- `data/val/cow_002/` - Cow 2 face crops for validation
- `data/val/cow_003/` - Cow 3 face crops for validation

## 🎯 Data Collection Guidelines

### For Perfect Human vs Cow Distinction:

#### Human Face Requirements:
- 📸 **Minimum 100+ images per person**
- 👤 **Multiple angles** (front, profile, 3/4 view)  
- 💡 **Different lighting** (indoor, outdoor, bright, dim)
- 🎭 **Various expressions** (neutral, smiling, serious)
- 👕 **Different clothing/backgrounds**
- 📏 **High resolution** (min 224x224 after cropping)

#### Cow Face Requirements:
- 📸 **Minimum 100+ images per cow**
- 🐄 **Multiple angles** (front, profile, 3/4 view)
- 💡 **Different lighting** (barn, field, sunny, cloudy)
- 📐 **Clear face visibility** (no obstructions)  
- 🏷️ **Consistent cow identity** (same cow in folder)
- 📏 **High resolution** (min 224x224 after cropping)

## 🔧 Processing Steps:

1. **Collect Raw Images**: Put images in raw/ folders
2. **Run Face Detection**: `python prepare_real_data.py --detect`
3. **Manual Review**: Check detected faces for quality
4. **Train Enhanced Model**: `python train_enhanced_model.py`
5. **Test Perfect Accuracy**: `python test_real_accuracy.py`

## 📊 Minimum Data Requirements for Production:

- **Humans**: 500-1000+ face crops (multiple people)
- **Cows**: 500-1000+ face crops (3-5 different cows)
- **80/20 train/val split** automatically applied
- **Balanced classes** for optimal training

## 🎉 Expected Results:

With proper real data:
- **>98% accuracy** on human vs cow distinction
- **Perfect production performance**
- **No confusion between species**
"""
        
        with open("data/DATA_ORGANIZATION_GUIDE.md", "w", encoding='utf-8') as f:
            f.write(guide_content)
        
        print(f"\n📖 Created data organization guide: data/DATA_ORGANIZATION_GUIDE.md")

    def detect_and_crop_faces(self, input_dir, output_dir, class_name):
        """Detect and crop faces from raw images"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            print(f"❌ Input directory not found: {input_path}")
            return 0
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if len(image_files) == 0:
            print(f"⚠️  No images found in {input_path}")
            return 0
        
        print(f"\n🔍 Processing {len(image_files)} images for {class_name}")
        
        face_count = 0
        output_path.mkdir(parents=True, exist_ok=True)
        
        for image_file in tqdm(image_files, desc=f"Processing {class_name}"):
            try:
                # Read image
                image = cv2.imread(str(image_file))
                if image is None:
                    continue
                
                # Detect faces
                detections = self.detector.detect(image)
                
                if len(detections) == 0:
                    print(f"⚠️  No faces detected in {image_file.name}")
                    continue
                
                # Process each detected face
                for idx, detection in enumerate(detections):
                    x1, y1, x2, y2, confidence = detection
                    
                    # Skip low confidence detections
                    if confidence < 0.5:
                        continue
                    
                    # Crop face with padding
                    face_crop = self.detector.crop_with_padding(
                        image, (x1, y1, x2, y2), pad_ratio=0.15
                    )
                    
                    # Resize to standard size
                    face_crop_resized = cv2.resize(face_crop, (224, 224))
                    
                    # Save cropped face
                    face_filename = f"{class_name}_{image_file.stem}_{idx}_{face_count:04d}.jpg"
                    face_path = output_path / face_filename
                    
                    cv2.imwrite(str(face_path), face_crop_resized)
                    face_count += 1
                    
            except Exception as e:
                print(f"⚠️  Error processing {image_file.name}: {str(e)}")
                continue
        
        print(f"✅ Extracted {face_count} faces for {class_name}")
        return face_count

    def split_train_val(self, class_dir, train_ratio=0.8):
        """Split data into training and validation sets"""
        class_path = Path(f"data/processed/{class_dir}")
        train_path = Path(f"data/train/{class_dir}")
        val_path = Path(f"data/val/{class_dir}")
        
        if not class_path.exists():
            return
        
        # Get all face images
        face_files = list(class_path.glob("*.jpg"))
        
        if len(face_files) == 0:
            return
        
        # Shuffle and split
        import random
        random.shuffle(face_files)
        
        split_idx = int(len(face_files) * train_ratio)
        train_files = face_files[:split_idx]
        val_files = face_files[split_idx:]
        
        # Create directories
        train_path.mkdir(parents=True, exist_ok=True)
        val_path.mkdir(parents=True, exist_ok=True)
        
        # Copy files
        for file in train_files:
            shutil.copy2(file, train_path / file.name)
            
        for file in val_files:
            shutil.copy2(file, val_path / file.name)
        
        print(f"📊 {class_dir}: {len(train_files)} train, {len(val_files)} val")

    def process_all_data(self):
        """Process all raw data into training format"""
        print("🚀 Processing all raw data for perfect accuracy training...")
        
        # Create temporary processed directory
        os.makedirs("data/processed", exist_ok=True)
        
        total_faces = 0
        
        # Process humans
        human_faces = self.detect_and_crop_faces(
            "data/raw/humans", 
            "data/processed/human", 
            "human"
        )
        total_faces += human_faces
        
        # Process cows - split into individual cow identities if possible
        # For now, we'll process all cows together and manually separate later
        cow_faces = self.detect_and_crop_faces(
            "data/raw/cows", 
            "data/processed/cow_mixed", 
            "cow"
        )
        total_faces += cow_faces
        
        if total_faces == 0:
            print("❌ No faces detected! Please:")
            print("   1. Add images to data/raw/humans/ and data/raw/cows/")
            print("   2. Ensure images contain clear, visible faces")
            print("   3. Check image formats (jpg, png, etc.)")
            return
        
        print(f"\n✅ Total faces extracted: {total_faces}")
        
        # Split data
        print("\n📊 Splitting data into train/validation sets...")
        self.split_train_val("human")
        
        # Manual cow separation needed
        print(f"\n⚠️  MANUAL STEP REQUIRED:")
        print(f"📁 Please manually organize cow faces from data/processed/cow_mixed/")
        print(f"   into separate cow identity folders:")
        print(f"   - data/train/cow_001/ (Cow #1 faces)")  
        print(f"   - data/train/cow_002/ (Cow #2 faces)")
        print(f"   - data/train/cow_003/ (Cow #3 faces)")
        print(f"   - data/val/cow_001/ (Cow #1 validation)")
        print(f"   - data/val/cow_002/ (Cow #2 validation)")
        print(f"   - data/val/cow_003/ (Cow #3 validation)")

    def analyze_data_quality(self):
        """Analyze prepared data quality"""
        print("\n📊 ANALYZING DATA QUALITY FOR PERFECT ACCURACY")
        print("=" * 60)
        
        analysis = {}
        
        # Check training data
        train_path = Path("data/train")
        val_path = Path("data/val")
        
        for split_name, split_path in [("Training", train_path), ("Validation", val_path)]:
            print(f"\n{split_name} Data:")
            split_analysis = {}
            
            if split_path.exists():
                for class_dir in split_path.iterdir():
                    if class_dir.is_dir():
                        image_count = len(list(class_dir.glob("*.jpg")))
                        split_analysis[class_dir.name] = image_count
                        
                        # Quality assessment
                        if image_count >= 100:
                            quality = "✅ EXCELLENT"
                        elif image_count >= 50:
                            quality = "👍 GOOD"
                        elif image_count >= 20:
                            quality = "⚠️  MINIMUM"
                        else:
                            quality = "❌ INSUFFICIENT"
                        
                        print(f"   {class_dir.name}: {image_count} images {quality}")
            
            analysis[split_name.lower()] = split_analysis
        
        # Overall assessment
        total_humans = analysis.get('training', {}).get('human', 0)
        total_cows = sum(count for class_name, count in analysis.get('training', {}).items() 
                        if class_name.startswith('cow'))
        
        print(f"\n🎯 OVERALL ASSESSMENT:")
        print(f"   Total Human Faces: {total_humans}")
        print(f"   Total Cow Faces: {total_cows}")
        
        if total_humans >= 200 and total_cows >= 200:
            print("🎉 EXCELLENT! Dataset ready for >98% accuracy training!")
        elif total_humans >= 100 and total_cows >= 100:
            print("👍 GOOD! Dataset should achieve >95% accuracy")
        elif total_humans >= 50 and total_cows >= 50:
            print("⚠️  MINIMUM dataset. Expect ~90% accuracy")
        else:
            print("❌ INSUFFICIENT data for reliable training")
            print("💡 Recommendations:")
            print("   - Collect at least 100+ images per class")
            print("   - Ensure diverse poses, lighting, backgrounds")
            print("   - Maintain clear face visibility")
        
        # Save analysis
        with open("data/data_quality_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)
        
        print(f"\n📄 Analysis saved to: data/data_quality_analysis.json")

def main():
    parser = argparse.ArgumentParser(description="Prepare real data for perfect accuracy training")
    parser.add_argument("--setup", action="store_true", 
                       help="Create directory structure and guide")
    parser.add_argument("--detect", action="store_true",
                       help="Detect and crop faces from raw images")
    parser.add_argument("--analyze", action="store_true",
                       help="Analyze data quality for training")
    parser.add_argument("--all", action="store_true",
                       help="Run complete data preparation pipeline")
    
    args = parser.parse_args()
    
    preparator = RealDataPreparator()
    
    if args.setup or args.all:
        preparator.create_data_structure()
    
    if args.detect or args.all:
        preparator.process_all_data()
    
    if args.analyze or args.all:
        preparator.analyze_data_quality()
    
    if not any(vars(args).values()):
        print("📋 REAL DATA PREPARATION OPTIONS:")
        print("   --setup     Create directory structure")
        print("   --detect    Process raw images into face crops")
        print("   --analyze   Analyze data quality")
        print("   --all       Run complete preparation pipeline")
        print()
        print("🚀 Quick Start:")
        print("1. python prepare_real_data.py --setup")
        print("2. Add your images to data/raw/humans/ and data/raw/cows/")
        print("3. python prepare_real_data.py --detect")
        print("4. Manually organize cow faces by individual cow")
        print("5. python prepare_real_data.py --analyze")
        print("6. python train_enhanced_model.py")

if __name__ == "__main__":
    main()
