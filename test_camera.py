#!/usr/bin/env python3
"""
Simple camera test to verify camera functionality
"""

import cv2
import sys
import time

def test_camera(camera_id=0):
    """Test camera functionality"""
    print(f"🎥 Testing camera {camera_id}...")
    
    # Try to open camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"❌ Cannot open camera {camera_id}")
        print("💡 Try different camera IDs (0, 1, 2, etc.)")
        return False
    
    print(f"✅ Camera {camera_id} opened successfully!")
    print("📹 Press 'q' to quit, 's' to save frame")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Cannot read frame")
            break
        
        frame_count += 1
        
        # Add frame info
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 's' to save frame", (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Camera Test', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"test_frame_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"💾 Frame saved as {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("📹 Camera test completed!")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        camera_id = int(sys.argv[1])
    else:
        camera_id = 0
    
    test_camera(camera_id)
