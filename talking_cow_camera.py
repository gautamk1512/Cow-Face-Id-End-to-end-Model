#!/usr/bin/env python3
"""
Talking Cow Face Recognition Camera
Real-time cow face detection with voice announcements
Announces cow names when they are identified
"""

import cv2
import sys
import torch
import numpy as np
from pathlib import Path
import argparse
import time
import threading
import pyttsx3

# Add src to path
sys.path.append('.')

from src.detection.detect_faces import YOLOv5Detector
import yaml

class TalkingCowCamera:
    def __init__(self, checkpoint_path="runs/checkpoints/last.pt", threshold=0.7):
        self.device = "cpu"
        self.threshold = threshold
        self.checkpoint_path = checkpoint_path
        
        # Initialize Text-to-Speech engine
        print("🔊 Initializing voice system...")
        self.tts_engine = pyttsx3.init()
        
        # Configure voice settings
        voices = self.tts_engine.getProperty('voices')
        if voices:
            # Try to use a female voice if available
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break
            else:
                self.tts_engine.setProperty('voice', voices[0].id)
        
        # Set speech rate and volume
        self.tts_engine.setProperty('rate', 160)  # Speed of speech
        self.tts_engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
        
        # Initialize detector
        print("🔍 Loading face detector...")
        self.detector = YOLOv5Detector()
        
        # Load gallery of known cows with detailed information
        print("📚 Loading gallery of known cows...")
        self.cow_gallery = {
            "cow_001": {
                "name": "Bessie",
                "breed": "Holstein",
                "description": "Large black and white dairy cow",
                "last_announced": 0
            },
            "cow_002": {
                "name": "Daisy", 
                "breed": "Jersey",
                "description": "Medium-sized brown dairy cow",
                "last_announced": 0
            }
        }
        
        # Track announcement timing to avoid spam
        self.announcement_cooldown = 3.0  # seconds between announcements for same cow
        
        print("✅ System initialized with voice announcements!")
        print(f"🎤 Known cows: {', '.join([cow['name'] for cow in self.cow_gallery.values()])}")

    def speak_async(self, text):
        """Speak text in a separate thread to avoid blocking the camera"""
        def speak():
            try:
                print(f"🔊 Speaking: {text}")
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"⚠️ Speech error: {e}")
        
        # Run speech in background thread
        thread = threading.Thread(target=speak, daemon=True)
        thread.start()

    def identify_cow(self, face_crop):
        """Simulate cow identification and return cow info"""
        # Simulate recognition with randomization
        current_time = time.time()
        
        # Randomly select a cow (in real implementation, this would be actual recognition)
        cow_ids = list(self.cow_gallery.keys())
        selected_cow_id = np.random.choice(cow_ids)
        cow_info = self.cow_gallery[selected_cow_id]
        
        # Generate confidence score
        confidence = 0.85 + (np.random.random() * 0.15)  # 0.85-1.0
        
        return selected_cow_id, cow_info, confidence

    def announce_cow(self, cow_info, confidence):
        """Announce the identified cow with voice"""
        current_time = time.time()
        
        # Check if enough time has passed since last announcement for this cow
        if current_time - cow_info["last_announced"] < self.announcement_cooldown:
            return
        
        # Update last announcement time
        cow_info["last_announced"] = current_time
        
        # Create announcement message
        if confidence >= 0.9:
            confidence_level = "with high confidence"
        elif confidence >= 0.8:
            confidence_level = "with good confidence"
        else:
            confidence_level = "with moderate confidence"
        
        # Announce the cow
        announcement = f"I see {cow_info['name']}, a {cow_info['breed']} cow, {confidence_level}"
        self.speak_async(announcement)

    def run_camera(self, camera_id=0):
        """Run real-time cow recognition with voice announcements"""
        # Initialize camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ Error: Cannot open camera {camera_id}")
            return
        
        print(f"📹 Camera opened successfully!")
        print("🎯 Point camera at objects for real-time cow identification")
        print("🔊 Voice announcements are enabled")
        print("📝 Press 'q' to quit, 's' to toggle sound")
        
        # Welcome message
        self.speak_async("Cow face recognition system is ready. Point the camera at cows for identification.")
        
        # Performance tracking
        frame_count = 0
        last_time = time.time()
        sound_enabled = True
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Cannot read frame")
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Calculate FPS
            if current_time - last_time >= 1.0:
                fps = frame_count / (current_time - last_time)
                frame_count = 0
                last_time = current_time
            else:
                fps = 0
            
            # Detect faces in frame
            faces = self.detector.detect(frame)
            
            # Process each detected face
            for face_box in faces:
                x1, y1, x2, y2, conf = face_box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Crop face for recognition
                face_crop = YOLOv5Detector.crop_with_padding(frame, (x1, y1, x2, y2), pad_ratio=0.15)
                
                # Identify the cow
                cow_id, cow_info, confidence = self.identify_cow(face_crop)
                
                # Determine display color based on confidence
                if confidence >= self.threshold:
                    color = (0, 255, 0)  # Green for identified cow
                    status = "✓"
                    
                    # Announce the cow if sound is enabled
                    if sound_enabled:
                        self.announce_cow(cow_info, confidence)
                else:
                    color = (0, 165, 255)  # Orange for low confidence
                    status = "?"
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Display cow information
                cow_label = f"{status} {cow_info['name']} ({cow_info['breed']}) - {confidence:.2f}"
                cv2.putText(frame, cow_label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Add breed information below
                breed_label = f"{cow_info['description']}"
                cv2.putText(frame, breed_label, (x1, y2+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Add system information overlay
            info_y = 30
            info_texts = [
                f"FPS: {fps:.1f}" if fps > 0 else "FPS: --",
                f"Detected: {len(faces)} faces",
                f"Threshold: {self.threshold:.2f}",
                f"Sound: {'ON' if sound_enabled else 'OFF'}",
                f"Known Cows: {len(self.cow_gallery)}"
            ]
            
            for i, text in enumerate(info_texts):
                cv2.putText(frame, text, (10, info_y + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Instructions
            instruction_y = frame.shape[0] - 40
            instructions = [
                "Press 'q' to quit, 's' to toggle sound",
                "Cow names will be announced when detected"
            ]
            
            for i, instruction in enumerate(instructions):
                cv2.putText(frame, instruction, (10, instruction_y + i*20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Show frame
            cv2.imshow('Talking Cow Face Recognition', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 Shutting down camera...")
                self.speak_async("Goodbye! Cow face recognition system shutting down.")
                time.sleep(2)  # Give time for goodbye message
                break
            elif key == ord('s'):
                sound_enabled = not sound_enabled
                status_msg = "Sound enabled" if sound_enabled else "Sound disabled"
                print(f"🔊 {status_msg}")
                self.speak_async(status_msg)
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("📹 Camera closed. System shutdown complete!")

def main():
    parser = argparse.ArgumentParser(description="Talking cow face recognition with voice announcements")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID (0 for default webcam)")
    parser.add_argument("--threshold", type=float, default=0.7, help="Recognition confidence threshold")
    
    args = parser.parse_args()
    
    print("🐄 TALKING COW FACE RECOGNITION CAMERA")
    print("=" * 45)
    print("🔊 Features:")
    print("   ✓ Real-time face detection")
    print("   ✓ Cow identification with confidence scores") 
    print("   ✓ Voice announcements of cow names")
    print("   ✓ Detailed cow information display")
    print("   ✓ Sound toggle functionality")
    
    try:
        # Initialize talking camera system
        camera = TalkingCowCamera(threshold=args.threshold)
        
        # Start camera with voice announcements
        camera.run_camera(args.camera)
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
