#!/usr/bin/env python3
"""
Enhanced Cow Face Recognition with Speech Output
Real-time recognition with voice announcement of identified cows
"""

import cv2
import sys
import torch
import numpy as np
from pathlib import Path
import argparse
import time
import pyttsx3
import threading
from queue import Queue

# Add src to path
sys.path.append('src')

from src.models.vit_arcface import ViTArcFace
from src.detection.detect_faces import YOLOv5Detector
from src.datasets.cowface_dataset import CowFaceDataset
import yaml

class EnhancedCowRecognition:
    def __init__(self, checkpoint_path, gallery_dir, config_path="configs/default.yaml", threshold=0.5):
        self.threshold = threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("🎤 Initializing Text-to-Speech engine...")
        # Initialize speech engine
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)  # Speed of speech
            self.tts_engine.setProperty('volume', 0.9)  # Volume level
            
            # Get available voices
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Try to use a female voice if available
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            
            self.speech_queue = Queue()
            self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
            self.speech_thread.start()
            print("✅ Speech engine ready!")
            
        except Exception as e:
            print(f"⚠️  Speech engine failed to initialize: {e}")
            self.tts_engine = None
        
        # Load config with error handling
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            print("✅ Configuration loaded")
        except Exception as e:
            print(f"❌ Config error: {e}")
            # Use default config
            self.config = {
                'model': {'vit_name': 'vit_base_patch16_224', 'embed_dim': 512},
                'input': {'img_size': 224, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
            }
        
        # Initialize face detector with error handling
        print("🔍 Loading face detector...")
        try:
            self.detector = YOLOv5Detector()
            print("✅ Face detector ready")
        except Exception as e:
            print(f"❌ Face detector error: {e}")
            return
        
        # Load trained model with error handling
        print("🤖 Loading cow recognition model...")
        try:
            self.model = self._load_model(checkpoint_path)
            print("✅ Recognition model ready")
        except Exception as e:
            print(f"❌ Model loading error: {e}")
            return
        
        # Build gallery with error handling
        print("📁 Building cow gallery...")
        try:
            self.gallery_embeddings, self.gallery_labels, self.gallery_files = self._build_gallery(gallery_dir)
            print(f"✅ Gallery ready with {len(self.gallery_labels)} known faces")
            
            # Debug: Show what classes are in the gallery
            unique_labels = list(set(self.gallery_labels))
            print(f"🏷️  Available classes in gallery: {unique_labels}")
            for label in unique_labels:
                count = self.gallery_labels.count(label)
                if "human" in label.lower():
                    print(f"   👤 {label}: {count} samples")
                elif "cow" in label.lower():
                    print(f"   🐄 {label}: {count} samples")
                else:
                    print(f"   ❓ {label}: {count} samples")
                    
        except Exception as e:
            print(f"❌ Gallery building error: {e}")
            return
        
        # Track last spoken cow to avoid repetition
        self.last_spoken_cow = None
        self.last_spoken_time = 0
        self.speech_cooldown = 3.0  # seconds
        
        print(f"🎉 Enhanced system ready! Gallery contains {len(self.gallery_labels)} known faces")
        
    def _speech_worker(self):
        """Background thread for speech synthesis"""
        while True:
            try:
                message = self.speech_queue.get()
                if message is None:
                    break
                if self.tts_engine:
                    self.tts_engine.say(message)
                    self.tts_engine.runAndWait()
                self.speech_queue.task_done()
            except Exception as e:
                print(f"Speech error: {e}")
    
    def speak(self, message):
        """Add message to speech queue"""
        current_time = time.time()
        # Avoid repeating the same cow name too quickly
        if (self.last_spoken_cow != message or 
            current_time - self.last_spoken_time > self.speech_cooldown):
            
            if self.tts_engine:
                self.speech_queue.put(message)
                self.last_spoken_cow = message
                self.last_spoken_time = current_time
                print(f"🎤 Speaking: {message}")
    
    def _load_model(self, checkpoint_path):
        """Load model with comprehensive error handling"""
        try:
            # Get number of classes from gallery
            gallery_path = Path("data/val")
            if not gallery_path.exists():
                gallery_path = Path("data/train")
            
            if not gallery_path.exists():
                raise Exception("No training data found!")
            
            num_classes = len([d for d in gallery_path.iterdir() if d.is_dir()])
            print(f"📊 Found {num_classes} classes")
            
            # Initialize model
            model = ViTArcFace(
                vit_name=self.config['model']['vit_name'],
                num_classes=num_classes,
                embed_dim=self.config['model']['embed_dim'],
                pretrained=False
            )
            
            # Load checkpoint
            if not Path(checkpoint_path).exists():
                raise Exception(f"Checkpoint not found: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if 'model_state_dict' not in checkpoint:
                raise Exception("Invalid checkpoint format!")
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            return model.to(self.device)
            
        except Exception as e:
            raise Exception(f"Model loading failed: {e}")
    
    def _build_gallery(self, gallery_dir):
        """Build gallery with error handling"""
        try:
            if not Path(gallery_dir).exists():
                raise Exception(f"Gallery directory not found: {gallery_dir}")
            
            dataset = CowFaceDataset(
                root_dir=gallery_dir,
                img_size=self.config['input']['img_size'],
                is_train=False
            )
            
            if len(dataset) == 0:
                raise Exception("No images found in gallery!")
            
            embeddings = []
            labels = []
            files = []
            
            with torch.no_grad():
                for i in range(len(dataset)):
                    try:
                        img, label = dataset[i]
                        img_batch = img.unsqueeze(0).to(self.device)
                        
                        embedding = self.model.get_embedding(img_batch)
                        embeddings.append(embedding.cpu().numpy())
                        labels.append(dataset.classes[label])
                        files.append(dataset.image_paths[i])
                        
                    except Exception as e:
                        print(f"⚠️  Skipping gallery image {i}: {e}")
                        continue
            
            if len(embeddings) == 0:
                raise Exception("No valid embeddings created!")
            
            return np.vstack(embeddings), labels, files
            
        except Exception as e:
            raise Exception(f"Gallery building failed: {e}")
    
    def _preprocess_face(self, face_crop):
        """Preprocess detected face with error handling"""
        try:
            if face_crop is None or face_crop.size == 0:
                raise Exception("Invalid face crop")
            
            # Resize to model input size
            face_resized = cv2.resize(face_crop, (224, 224))
            
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            # Normalize
            face_normalized = face_rgb.astype(np.float32) / 255.0
            mean = np.array(self.config['input']['mean'])
            std = np.array(self.config['input']['std'])
            face_normalized = (face_normalized - mean) / std
            
            # Convert to tensor with correct dtype
            face_tensor = torch.from_numpy(face_normalized.transpose(2, 0, 1)).float().unsqueeze(0)
            return face_tensor.to(self.device)
            
        except Exception as e:
            raise Exception(f"Face preprocessing failed: {e}")
    
    def recognize_face(self, face_crop):
        """Recognize face with comprehensive error handling"""
        try:
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
            
            return best_label, best_similarity
            
        except Exception as e:
            print(f"Recognition error: {e}")
            return "Error", 0.0
    
    def run_camera(self, camera_id=0):
        """Run enhanced camera recognition with speech"""
        try:
            # Initialize camera
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                print(f"❌ Error: Cannot open camera {camera_id}")
                return
            
            # Welcome message
            self.speak("Cow Face Recognition System Ready")
            
            print(f"📹 Camera opened successfully! Press 'q' to quit")
            print("🎯 Point camera at faces for real-time recognition with speech")
            print("🎤 Speech announcements enabled!")
            
            # Get camera properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            if fps == 0:
                fps = 30
            
            frame_count = 0
            last_time = time.time()
            error_count = 0
            max_errors = 10
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    error_count += 1
                    if error_count > max_errors:
                        print("❌ Too many camera errors, stopping")
                        break
                    continue
                
                error_count = 0  # Reset error count on successful frame
                frame_count += 1
                current_time = time.time()
                
                # Calculate FPS
                if current_time - last_time >= 1.0:
                    current_fps = frame_count / (current_time - last_time)
                    frame_count = 0
                    last_time = current_time
                else:
                    current_fps = 0
                
                try:
                    # Detect faces in frame
                    boxes = self.detector.detect(frame)
                    
                    # Process each detected face
                    for box in boxes:
                        x1, y1, x2, y2, conf = box
                        
                        try:
                            # Draw detection box
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            
                            # Crop face with padding
                            face_crop = YOLOv5Detector.crop_with_padding(frame, (x1, y1, x2, y2), pad_ratio=0.15)
                            
                            # Recognize the face
                            cow_id, similarity = self.recognize_face(face_crop)
                            
                            # Choose color and status based on identity type
                            if similarity >= self.threshold and cow_id != "Error":
                                status = "✓"
                                
                                # Debug output to see what's being detected
                                print(f"🔍 Detected: '{cow_id}' with similarity {similarity:.3f}")
                                
                                # Different colors for humans vs cows - improved logic
                                if "human" in cow_id.lower() or cow_id.lower() == "human":
                                    color = (255, 0, 0)  # BLUE for humans (BGR format)
                                    speech_text = "Human detected"
                                    label_prefix = "HUMAN"
                                    print(f"👤 HUMAN detected: {cow_id}")
                                elif "cow" in cow_id.lower() or any(c in cow_id.lower() for c in ['cow_', 'bovine', 'cattle']):
                                    color = (0, 255, 0)  # GREEN for cows (BGR format)
                                    speech_text = f"Cow {cow_id} identified"
                                    label_prefix = "COW"
                                    print(f"🐄 COW detected: {cow_id}")
                                else:
                                    # If we can't clearly identify, treat as unknown for now
                                    color = (0, 165, 255)  # Orange for unclear classification
                                    speech_text = f"Entity {cow_id} detected"
                                    label_prefix = "ENTITY"
                                    print(f"❓ UNCLEAR classification: {cow_id}")
                                
                                self.speak(speech_text)
                                
                            elif cow_id == "Error":
                                color = (0, 0, 255)  # Red for error
                                status = "⚠"
                                label_prefix = "ERROR"
                            else:
                                color = (0, 165, 255)  # Orange for unknown
                                status = "?"
                                label_prefix = "UNKNOWN"
                                
                                if similarity > 0.3:  # Some confidence but below threshold
                                    self.speak("Unknown face detected")
                            
                            # Display result with clear labeling
                            if cow_id != "Error" and similarity >= self.threshold:
                                if cow_id.lower() == "human":
                                    label = f"HUMAN: {cow_id} ({similarity:.3f})"
                                else:
                                    label = f"COW: {cow_id} ({similarity:.3f})"
                            else:
                                label = f"{status} {cow_id} ({similarity:.3f})"
                            
                            cv2.putText(frame, label, (int(x1), int(y1-10)), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
                            # Update box color
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            
                        except Exception as e:
                            # Handle individual face processing errors
                            cv2.putText(frame, f"Face Error: {str(e)[:20]}", (int(x1), int(y1-10)), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                except Exception as e:
                    # Handle detection errors
                    cv2.putText(frame, f"Detection Error: {str(e)[:30]}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    boxes = []
                
                # Add enhanced info overlay
                info_text = [
                    f"FPS: {current_fps:.1f}" if current_fps > 0 else "FPS: --",
                    f"Faces: {len(boxes) if 'boxes' in locals() else 0}",
                    f"Threshold: {self.threshold:.2f}",
                    f"Gallery: {len(self.gallery_labels)} identities",
                    f"Speech: {'ON' if self.tts_engine else 'OFF'}",
                    f"Device: {str(self.device).upper()}"
                ]
                
                for i, text in enumerate(info_text):
                    cv2.putText(frame, text, (10, 30 + i*25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Color Legend - Top Right Corner
                legend_start_x = frame.shape[1] - 280
                legend_start_y = 30
                
                # Legend background
                cv2.rectangle(frame, (legend_start_x - 10, legend_start_y - 5), 
                             (legend_start_x + 270, legend_start_y + 85), (0, 0, 0), -1)
                cv2.rectangle(frame, (legend_start_x - 10, legend_start_y - 5), 
                             (legend_start_x + 270, legend_start_y + 85), (255, 255, 255), 2)
                
                # Legend title
                cv2.putText(frame, "COLOR LEGEND:", (legend_start_x, legend_start_y + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Human - Blue
                cv2.rectangle(frame, (legend_start_x, legend_start_y + 25), 
                             (legend_start_x + 20, legend_start_y + 35), (255, 0, 0), -1)
                cv2.putText(frame, "HUMAN", (legend_start_x + 30, legend_start_y + 33), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
                
                # Cow - Green
                cv2.rectangle(frame, (legend_start_x, legend_start_y + 45), 
                             (legend_start_x + 20, legend_start_y + 55), (0, 255, 0), -1)
                cv2.putText(frame, "COW", (legend_start_x + 30, legend_start_y + 53), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
                
                # Unknown - Orange
                cv2.rectangle(frame, (legend_start_x, legend_start_y + 65), 
                             (legend_start_x + 20, legend_start_y + 75), (0, 165, 255), -1)
                cv2.putText(frame, "UNKNOWN", (legend_start_x + 30, legend_start_y + 73), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 2)
                
                # Enhanced instructions
                instructions = [
                    "Press 'q' to quit",
                    "Press 's' to toggle speech",
                    "Press 'r' to reset system"
                ]
                
                for i, instruction in enumerate(instructions):
                    cv2.putText(frame, instruction, (10, frame.shape[0] - 60 + i*20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Show frame
                cv2.imshow('Enhanced Cow Face Recognition - Real Time', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.speak("Goodbye")
                    break
                elif key == ord('s'):
                    # Toggle speech
                    if self.tts_engine:
                        self.speak("Speech toggled")
                    print("🎤 Speech toggle requested")
                elif key == ord('r'):
                    print("🔄 System reset requested")
                    self.speak("System reset")
            
        except Exception as e:
            print(f"❌ Camera error: {e}")
            if self.tts_engine:
                self.speak("Camera error occurred")
        
        finally:
            # Cleanup
            try:
                cap.release()
                cv2.destroyAllWindows()
                print("📹 Camera closed. System shutdown complete!")
                
                # Stop speech thread
                if self.tts_engine:
                    self.speech_queue.put(None)
                    
            except Exception as e:
                print(f"Cleanup error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced cow face recognition with speech")
    parser.add_argument("--checkpoint", type=str, default="runs/enhanced_checkpoints/best_model.pt", 
                       help="Path to trained model checkpoint")
    parser.add_argument("--gallery", type=str, default="data/val", 
                       help="Path to gallery of known faces")
    parser.add_argument("--camera", type=int, default=0, 
                       help="Camera ID (0 for default webcam)")
    parser.add_argument("--threshold", type=float, default=0.5, 
                       help="Recognition confidence threshold")
    parser.add_argument("--config", type=str, default="configs/default.yaml", 
                       help="Path to config file")
    
    args = parser.parse_args()
    
    print("🐄 ENHANCED COW FACE RECOGNITION WITH SPEECH")
    print("=" * 50)
    print("🎤 Features: Real-time recognition + Speech announcements")
    print()
    
    try:
        # Check if model exists
        if not Path(args.checkpoint).exists():
            print(f"❌ Model checkpoint not found: {args.checkpoint}")
            print("💡 Train a model first using: python simple_train_model.py")
            return
        
        # Initialize enhanced recognition system
        recognizer = EnhancedCowRecognition(
            checkpoint_path=args.checkpoint,
            gallery_dir=args.gallery,
            config_path=args.config,
            threshold=args.threshold
        )
        
        # Start enhanced camera recognition
        recognizer.run_camera(args.camera)
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ System Error: {str(e)}")
        print("💡 Check your installation and try again")

if __name__ == "__main__":
    main()
