#!/usr/bin/env python3
"""
Evaluate the cow face recognition model on the validation dataset
"""

import sys
import os
from pathlib import Path

sys.path.append('src')

def evaluate_validation_set():
    print("🔬 EVALUATING MODEL ON VALIDATION SET")
    print("=" * 45)
    
    val_dir = Path("data/val")
    model_path = "runs/checkpoints/best.pt"
    
    if not Path(model_path).exists():
        print("❌ Model not found!")
        return
    
    correct = 0
    total = 0
    results = []
    
    # Test each cow identity
    for cow_dir in val_dir.iterdir():
        if cow_dir.is_dir():
            cow_id = cow_dir.name
            print(f"\n📋 Testing {cow_id}:")
            
            for img_path in cow_dir.glob("*.jpg"):
                cmd = f'python -m src.inference.recognize --checkpoint {model_path} --gallery_dir data/val --query "{img_path}" --top_k 1 --threshold 0.35'
                
                try:
                    result = os.popen(cmd).read().strip()
                    if result:
                        result_dict = eval(result)
                        predicted = result_dict['predicted_label']
                        similarity = result_dict['best_similarity']
                        
                        is_correct = predicted == cow_id
                        status = "✅" if is_correct else "❌"
                        
                        print(f"   {img_path.name}: {predicted} (sim: {similarity:.4f}) {status}")
                        
                        if is_correct:
                            correct += 1
                        total += 1
                        
                        results.append({
                            'image': str(img_path),
                            'expected': cow_id,
                            'predicted': predicted,
                            'similarity': similarity,
                            'correct': is_correct
                        })
                except Exception as e:
                    print(f"   {img_path.name}: ERROR - {str(e)}")
    
    # Calculate accuracy
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print(f"\n📊 EVALUATION RESULTS:")
    print(f"   Total images tested: {total}")
    print(f"   Correct predictions: {correct}")
    print(f"   Accuracy: {accuracy:.2f}%")
    
    if accuracy >= 90:
        print("🎉 EXCELLENT MODEL PERFORMANCE!")
    elif accuracy >= 70:
        print("👍 GOOD MODEL PERFORMANCE!")
    else:
        print("⚠️  MODEL NEEDS IMPROVEMENT!")

if __name__ == "__main__":
    evaluate_validation_set()
