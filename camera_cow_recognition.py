#!/usr/bin/env python3
"""
Real-time Cow Face Recognition using Camera
Uses webcam or camera feed to detect and identify cow faces in real-time
"""

import cv2
import sys
import torch
import numpy as np
from pathlib import Path
import argparse
import time

# Add src to path
sys.path.append('src')

from src.models.vit_arcface import ViTArcFace
from src.detection.detect_faces import YOLOv5Detector
from src.datasets.cowface_dataset import CowFaceDataset
import yaml

class RealTimeCowRecognition:
    def __init__(self, checkpoint_path, gallery_dir, config_path="configs/default.yaml", threshold=0.5):
        self.threshold = threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize face detector
        print("🔍 Loading face detector...")
        self.detector = YOLOv5Detector()
        
        # Load trained model
        print("🤖 Loading cow recognition model...")
        self.model = self._load_model(checkpoint_path)
        
        # Build gallery (known cow faces)
        print("📁 Building cow gallery...")
        self.gallery_embeddings, self.gallery_labels, self.gallery_files = self._build_gallery(gallery_dir)
        
        print(f"✅ System ready! Gallery contains {len(self.gallery_labels)} known cow faces")
        
    def _load_model(self, checkpoint_path):
        # Load checkpoint first to get the number of classes
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Get number of classes from checkpoint or config
        if 'classes' in checkpoint:
            num_classes = len(checkpoint['classes'])
        elif 'config' in checkpoint and 'num_classes' in checkpoint['config']:
            num_classes = checkpoint['config']['num_classes']
        else:
            # Fallback: count directories in gallery
            gallery_path = Path("data/val")
            num_classes = len([d for d in gallery_path.iterdir() if d.is_dir()])
        
        model = ViTArcFace(
            vit_name=self.config['model']['vit_name'],
            num_classes=num_classes,
            embed_dim=self.config['model']['embed_dim'],
            pretrained=False
        )
        
        # Handle state_dict key mismatch by adding 'model.' prefix if needed
        state_dict = checkpoint['model_state']
        
        # Check if we need to add 'model.' prefix
        if 'backbone.cls_token' in state_dict and 'model.backbone.cls_token' not in state_dict:
            # Add 'model.' prefix to all keys
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = f"model.{key}"
                new_state_dict[new_key] = value
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        model.eval()
        return model.to(self.device)
    
    def _build_gallery(self, gallery_dir):
        dataset = CowFaceDataset(
            root_dir=gallery_dir,
            img_size=self.config['input']['img_size'],
            mean=self.config['input']['mean'],
            std=self.config['input']['std'],
            is_train=False
        )
        
        embeddings = []
        labels = []
        files = []
        
        with torch.no_grad():
            for i in range(len(dataset)):
                img, label = dataset[i]
                img_batch = img.unsqueeze(0).to(self.device)
                
                embedding = self.model.get_embedding(img_batch)
                embeddings.append(embedding.cpu().numpy())
                labels.append(dataset.classes[label])
                files.append(dataset.image_paths[i])
        
        return np.vstack(embeddings), labels, files
    
    def _preprocess_face(self, face_crop):
        """Preprocess detected face for recognition"""
        # Resize to model input size
        face_resized = cv2.resize(face_crop, (224, 224))
        
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize
        face_normalized = face_rgb.astype(np.float32) / 255.0
        mean = np.array(self.config['input']['mean'])
        std = np.array(self.config['input']['std'])
        face_normalized = (face_normalized - mean) / std
        
        # Convert to tensor
        face_tensor = torch.from_numpy(face_normalized.transpose(2, 0, 1)).unsqueeze(0)
        return face_tensor.to(self.device)
    
    def recognize_face(self, face_crop):
        """Recognize a cow face and return the identity"""
        face_tensor = self._preprocess_face(face_crop)
        
        with torch.no_grad():
            embedding = self.model.get_embedding(face_tensor)
            embedding_np = embedding.cpu().numpy()
        
        # Calculate similarities with gallery
        similarities = np.dot(self.gallery_embeddings, embedding_np.T).flatten()
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        best_label = self.gallery_labels[best_idx]
        
        if best_similarity >= self.threshold:
            return best_label, best_similarity
        else:
            return "Unknown", best_similarity
    
    def run_camera(self, camera_id=0):
        """Run real-time recognition on camera feed"""
        # Initialize camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ Error: Cannot open camera {camera_id}")
            return
        
        print(f"📹 Camera opened successfully! Press 'q' to quit")
        print("🎯 Point camera at cow faces for real-time recognition")
        
        # Get camera properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30
        
        frame_count = 0
        last_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Cannot read frame")
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Calculate FPS
            if current_time - last_time >= 1.0:
                current_fps = frame_count / (current_time - last_time)
                frame_count = 0
                last_time = current_time
            else:
                current_fps = 0
            
            # Detect faces in frame
            boxes = self.detector.detect(frame)
            
            # Process each detected face
            for box in boxes:
                x1, y1, x2, y2, conf = box
                
                # Draw detection box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Crop face with padding
                face_crop = YOLOv5Detector.crop_with_padding(frame, (x1, y1, x2, y2), pad_ratio=0.15)
                
                # Recognize the face
                try:
                    cow_id, similarity = self.recognize_face(face_crop)
                    
                    # Choose color based on confidence
                    if similarity >= self.threshold:
                        color = (0, 255, 0)  # Green for known cow
                        status = "✓"
                    else:
                        color = (0, 165, 255)  # Orange for unknown
                        status = "?"
                    
                    # Display result
                    label = f"{status} {cow_id} ({similarity:.3f})"
                    cv2.putText(frame, label, (int(x1), int(y1-10)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Update box color
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                except Exception as e:
                    # Handle recognition errors
                    cv2.putText(frame, f"Error: {str(e)[:20]}", (int(x1), int(y1-10)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Add info overlay
            info_text = [
                f"FPS: {current_fps:.1f}" if current_fps > 0 else "FPS: --",
                f"Faces: {len(boxes)}",
                f"Threshold: {self.threshold:.2f}",
                f"Gallery: {len(self.gallery_labels)} cows"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(frame, text, (10, 30 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Instructions
            cv2.putText(frame, "Press 'q' to quit, 'r' to reset", 
                       (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow('Cow Face Recognition - Real Time', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("🔄 Resetting recognition system...")
                continue
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("📹 Camera closed. Goodbye!")

def main():
    parser = argparse.ArgumentParser(description="Real-time cow face recognition using camera")
    parser.add_argument("--checkpoint", type=str, default="runs/checkpoints/best.pt", 
                       help="Path to trained model checkpoint")
    parser.add_argument("--gallery", type=str, default="data/val", 
                       help="Path to gallery of known cow faces")
    parser.add_argument("--camera", type=int, default=0, 
                       help="Camera ID (0 for default webcam)")
    parser.add_argument("--threshold", type=float, default=0.5, 
                       help="Recognition confidence threshold")
    parser.add_argument("--config", type=str, default="configs/default.yaml", 
                       help="Path to config file")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.checkpoint).exists():
        print(f"❌ Model checkpoint not found: {args.checkpoint}")
        print("💡 Train a model first using: python -m src.training.train_classifier")
        return
    
    print("🐄 REAL-TIME COW FACE RECOGNITION")
    print("=" * 40)
    
    try:
        # Initialize recognition system
        recognizer = RealTimeCowRecognition(
            checkpoint_path=args.checkpoint,
            gallery_dir=args.gallery,
            config_path=args.config,
            threshold=args.threshold
        )
        
        # Start camera recognition
        recognizer.run_camera(args.camera)
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
