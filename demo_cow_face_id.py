#!/usr/bin/env python3
"""
Demo script for Cow Face ID - End-to-end Model
Demonstrates the complete pipeline: Detection, Training, and Recognition
"""

import sys
import os
from pathlib import Path
import json

# Add src to path
sys.path.append('src')

def main():
    print("🐄 COW FACE ID - END-TO-END MODEL DEMO")
    print("=" * 50)
    
    # 1. Show project structure
    print("\n📁 PROJECT STRUCTURE:")
    print("✓ YOLOv5 for face detection")
    print("✓ Vision Transformer (ViT) for feature extraction")
    print("✓ ArcFace for similarity learning")
    print("✓ Training data: 16 images (2 cow identities)")
    print("✓ Validation data: 4 images")
    
    # 2. Check model availability
    model_path = Path("runs/checkpoints/last.pt")
    if model_path.exists():
        print(f"\n🤖 MODEL STATUS: ✓ Trained model available ({model_path})")
    else:
        print("\n🤖 MODEL STATUS: ❌ No trained model found")
        return
    
    # 3. Test recognition on different images
    print("\n🔍 TESTING COW RECOGNITION:")
    
    test_cases = [
        ("data/train/cow_001/train_cow_001_0.jpg", "cow_001"),
        ("data/train/cow_002/train_cow_002_0.jpg", "cow_002"),
        ("data/val/cow_001/val_cow_001_0.jpg", "cow_001"),
        ("data/val/cow_002/val_cow_002_0.jpg", "cow_002")
    ]
    
    for query_img, expected_cow in test_cases:
        if Path(query_img).exists():
            print(f"\n🎯 Testing: {query_img}")
            print(f"   Expected: {expected_cow}")
            
            # Run recognition command
            cmd = f'python -m src.inference.recognize --checkpoint runs/checkpoints/last.pt --gallery_dir data/val --query "{query_img}" --top_k 3 --threshold 0.35'
            try:
                result = os.popen(cmd).read().strip()
                if result:
                    # Parse the result
                    result_dict = eval(result)
                    predicted = result_dict['predicted_label']
                    similarity = result_dict['best_similarity']
                    
                    status = "✅ CORRECT" if predicted == expected_cow else "❌ INCORRECT"
                    print(f"   Predicted: {predicted} (similarity: {similarity:.4f}) {status}")
                else:
                    print("   ❌ No result returned")
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
    
    # 4. Show training capabilities
    print(f"\n📊 TRAINING CAPABILITIES:")
    print("✓ Configurable epochs, batch size, learning rate")
    print("✓ Data augmentation (horizontal flip, color jitter)")
    print("✓ Mixed precision training support")
    print("✓ Validation monitoring")
    print("✓ Best model checkpoint saving")
    
    # 5. Show detection capabilities
    print(f"\n🎯 DETECTION CAPABILITIES:")
    print("✓ YOLOv5-based cow face detection")
    print("✓ Automatic cropping with padding")
    print("✓ Batch processing of image directories")
    print("✓ Custom detector model support")
    
    print(f"\n🎉 DEMO COMPLETED!")
    print("🔧 Available commands:")
    print("   Training: python -m src.training.train_classifier --data_root data --config configs/default.yaml")
    print("   Recognition: python -m src.inference.recognize --checkpoint runs/checkpoints/last.pt --gallery_dir data/val --query path/to/image.jpg")
    print("   Detection: python -m scripts.detect_and_crop --input_dir raw_images --output_dir cropped_faces")

if __name__ == "__main__":
    main()
