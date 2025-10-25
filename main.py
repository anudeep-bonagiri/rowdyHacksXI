#!/usr/bin/env python3
"""
🤚 ARCHAND - Advanced Hand Gesture Recognition System
=====================================================

A hackathon-ready computer vision application that transforms your hand gestures
into computer control commands. Built with MediaPipe, OpenCV, and PyAutoGUI.

Features:
- 🖱️  Mouse control with hand tracking
- 👆  Multi-gesture recognition (click, drag, scroll)
- 🎤  Speech-to-text integration
- ⚡  Real-time performance with 60 FPS
- 🎨  Beautiful visual feedback
- 🔧  Configurable gesture sensitivity

Author: Hackathon Team
License: MIT
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import autopy
import time
import speech_recognition as sr
import argparse
import sys
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class GestureType(Enum):
    """Enumeration of supported gesture types"""
    MOVE = "move"
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    DRAG = "drag"
    SCROLL_UP = "scroll_up"
    SCROLL_DOWN = "scroll_down"
    SPEECH = "speech"


@dataclass
class Config:
    """Configuration class for the gesture recognition system"""
    # Camera settings
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 60
    
    # Gesture detection settings
    frame_region: int = 100
    smoothing_factor: int = 8
    stability_threshold: int = 10
    stability_radius: int = 10
    
    # Scroll settings
    scroll_up_speed: int = 60
    scroll_down_speed: int = -60
    
    # Speech settings
    speech_timeout: int = 5
    speech_phrase_limit: int = 10
    typing_interval: float = 0.01
    
    # Visual settings
    show_fps: bool = True
    show_gesture_info: bool = True


class HandGestureRecognizer:
    """
    Advanced hand gesture recognition system using MediaPipe
    
    This class handles all aspects of hand detection, landmark tracking,
    and gesture classification with real-time performance optimization.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.setup_mediapipe()
        self.setup_speech_recognition()
        self.setup_screen_info()
        
        # State tracking
        self.previous_x = 0
        self.previous_y = 0
        self.current_x = 0
        self.current_y = 0
        self.previous_time = 0
        self.stability_buffer = []
        self.drag_hold = False
        
        # Performance tracking
        self.frame_count = 0
        self.fps = 0
        
    def setup_mediapipe(self):
        """Initialize MediaPipe hand detection"""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Configure hand detection with optimized parameters
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Finger tip indices for gesture detection
        self.finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        
    def setup_speech_recognition(self):
        """Initialize speech recognition system"""
        try:
            self.speech_recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            print("🎤 Speech recognition initialized successfully")
        except Exception as e:
            print(f"⚠️  Speech recognition unavailable: {e}")
            self.speech_recognizer = None
            self.microphone = None
            
    def setup_screen_info(self):
        """Get screen dimensions for mouse mapping"""
        self.screen_width, self.screen_height = autopy.screen.size()
        print(f"🖥️  Screen resolution: {self.screen_width}x{self.screen_height}")
        
    def detect_hands(self, frame: np.ndarray, draw: bool = True) -> np.ndarray:
        """
        Detect hands in the frame using MediaPipe
        
        Args:
            frame: Input video frame
            draw: Whether to draw hand landmarks
            
        Returns:
            Processed frame with hand landmarks
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        self.results = self.hands.process(rgb_frame)
        
        # Draw landmarks if requested
        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
        return frame
        
    def get_hand_landmarks(self, frame: np.ndarray, hand_index: int = 0, draw: bool = True) -> Tuple[List, Tuple]:
        """
        Extract hand landmarks and bounding box
        
        Args:
            frame: Input video frame
            hand_index: Index of hand to track (0 for first hand)
            draw: Whether to draw landmarks
            
        Returns:
            Tuple of (landmarks, bounding_box)
        """
        landmarks = []
        bounding_box = (0, 0, 0, 0)
        
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_index]
            height, width, _ = frame.shape
            
            # Extract landmark coordinates
            x_coords, y_coords = [], []
            for idx, landmark in enumerate(hand.landmark):
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                landmarks.append([idx, x, y])
                x_coords.append(x)
                y_coords.append(y)
                
                # Draw landmark points
                if draw:
                    cv2.circle(frame, (x, y), 5, (255, 0, 255), cv2.FILLED)
                    
            # Calculate bounding box
            if x_coords and y_coords:
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                bounding_box = (x_min - 20, y_min - 20, x_max + 20, y_max + 20)
                
                # Draw bounding box
                if draw:
                    cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20),
                                  (0, 255, 0), 2)
                                  
        return landmarks, bounding_box
        
    def detect_finger_states(self, landmarks: List) -> List[int]:
        """
        Detect which fingers are raised
        
        Args:
            landmarks: List of hand landmarks
            
        Returns:
            List of finger states (1 = raised, 0 = down)
        """
        if len(landmarks) < 21:  # Need all 21 landmarks
            return [0] * 5
            
        fingers = []
        
        # Thumb (different logic due to orientation)
        if landmarks[self.finger_tips[0]][1] > landmarks[self.finger_tips[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
            
        # Other fingers (Index, Middle, Ring, Pinky)
        for i in range(1, 5):
            if landmarks[self.finger_tips[i]][2] < landmarks[self.finger_tips[i] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers
        
    def classify_gesture(self, fingers: List[int]) -> GestureType:
        """
        Classify gesture based on finger states
        
        Args:
            fingers: List of finger states
            
        Returns:
            Detected gesture type
        """
        # Speech to text (middle finger only)
        if fingers == [0, 0, 1, 0, 0]:
            return GestureType.SPEECH
            
        # Right click (thumb only)
        if fingers == [1, 0, 0, 0, 0]:
            return GestureType.RIGHT_CLICK
            
        # Double click (index + pinky)
        if fingers == [0, 1, 0, 0, 1]:
            return GestureType.DOUBLE_CLICK
            
        # Drag (all fingers except thumb)
        if fingers == [0, 1, 1, 1, 1]:
            return GestureType.DRAG
            
        # Click (index + middle)
        if fingers == [0, 1, 1, 0, 0]:
            return GestureType.CLICK
            
        # Move (index only)
        if fingers == [0, 1, 0, 0, 0]:
            return GestureType.MOVE
            
        # Scroll up (all fingers)
        if fingers == [1, 1, 1, 1, 1]:
            return GestureType.SCROLL_UP
            
        # Scroll down (no fingers)
        if fingers == [0, 0, 0, 0, 0]:
            return GestureType.SCROLL_DOWN
            
        return GestureType.MOVE  # Default to move
        
    def execute_gesture(self, gesture: GestureType, landmarks: List, frame: np.ndarray):
        """
        Execute the appropriate action based on gesture type
        
        Args:
            gesture: Type of gesture detected
            landmarks: Hand landmarks
            frame: Current video frame
        """
        if len(landmarks) < 8:  # Need at least index finger landmark
            return
            
        x, y = landmarks[8][1], landmarks[8][2]  # Index finger tip
        
        # Map camera coordinates to screen coordinates
        screen_x = np.interp(x, (self.config.frame_region, 
                                self.config.camera_width - self.config.frame_region), 
                           (0, self.screen_width))
        screen_y = np.interp(y, (self.config.frame_region, 
                                self.config.camera_height - self.config.frame_region), 
                           (0, self.screen_height))
        
        # Apply smoothing
        self.current_x = self.previous_x + (screen_x - self.previous_x) / self.config.smoothing_factor
        self.current_y = self.previous_y + (screen_y - self.previous_y) / self.config.smoothing_factor
        
        # Execute gesture-specific actions
        if gesture == GestureType.MOVE:
            self._handle_mouse_move()
            self._stop_drag()  # Stop drag if not in drag gesture
            
        elif gesture == GestureType.CLICK:
            self._handle_click(x, y, frame)
            self._stop_drag()
            
        elif gesture == GestureType.DOUBLE_CLICK:
            self._handle_double_click()
            self._stop_drag()
            
        elif gesture == GestureType.RIGHT_CLICK:
            self._handle_right_click(x, y, frame)
            self._stop_drag()
            
        elif gesture == GestureType.DRAG:
            self._handle_drag()
            
        elif gesture == GestureType.SCROLL_UP:
            self._handle_scroll_up()
            self._stop_drag()
            
        elif gesture == GestureType.SCROLL_DOWN:
            self._handle_scroll_down()
            self._stop_drag()
            
        elif gesture == GestureType.SPEECH:
            self._handle_speech_to_text()
            self._stop_drag()
            
        # Update previous positions
        self.previous_x, self.previous_y = self.current_x, self.current_y
        
    def _handle_mouse_move(self):
        """Handle mouse movement"""
        autopy.mouse.move(self.screen_width - self.current_x, self.current_y)
        
    def _handle_click(self, x: int, y: int, frame: np.ndarray):
        """Handle single click with stability check"""
        self.stability_buffer.append((self.current_x, self.current_y))
        if len(self.stability_buffer) > self.config.stability_threshold:
            self.stability_buffer.pop(0)
            
        if (len(self.stability_buffer) == self.config.stability_threshold and
            all(np.linalg.norm(np.array(pos) - np.array(self.stability_buffer[0])) < self.config.stability_radius
                for pos in self.stability_buffer)):
            cv2.circle(frame, (x, y), 15, (0, 255, 0), cv2.FILLED)
            autopy.mouse.click()
            self.stability_buffer.clear()
            
    def _handle_double_click(self):
        """Handle double click"""
        autopy.mouse.click()
        autopy.mouse.click()
        
    def _handle_right_click(self, x: int, y: int, frame: np.ndarray):
        """Handle right click with stability check"""
        self.stability_buffer.append((self.current_x, self.current_y))
        if len(self.stability_buffer) > self.config.stability_threshold:
            self.stability_buffer.pop(0)
            
        if (len(self.stability_buffer) == self.config.stability_threshold and
            all(np.linalg.norm(np.array(pos) - np.array(self.stability_buffer[0])) < self.config.stability_radius
                for pos in self.stability_buffer)):
            cv2.circle(frame, (x, y), 15, (0, 255, 0), cv2.FILLED)
            autopy.mouse.click(autopy.mouse.Button.RIGHT)
            self.stability_buffer.clear()
            
    def _handle_drag(self):
        """Handle drag operation"""
        if not self.drag_hold:
            autopy.mouse.toggle(down=True)
            self.drag_hold = True
        autopy.mouse.move(self.screen_width - self.current_x, self.current_y)
        
    def _stop_drag(self):
        """Stop drag operation"""
        if self.drag_hold:
            autopy.mouse.toggle(down=False)
            self.drag_hold = False
            
    def _handle_scroll_up(self):
        """Handle scroll up"""
        pyautogui.scroll(self.config.scroll_up_speed)
        
    def _handle_scroll_down(self):
        """Handle scroll down"""
        pyautogui.scroll(self.config.scroll_down_speed)
        
    def _handle_speech_to_text(self):
        """Handle speech to text conversion"""
        if not self.speech_recognizer or not self.microphone:
            return
            
        try:
            with self.microphone as source:
                self.speech_recognizer.adjust_for_ambient_noise(source)
                audio = self.speech_recognizer.listen(
                    source, 
                    timeout=self.config.speech_timeout,
                    phrase_time_limit=self.config.speech_phrase_limit
                )
                text = self.speech_recognizer.recognize_google(audio)
                if text:
                    pyautogui.write(text, interval=self.config.typing_interval)
                    print(f"🎤 Speech recognized: {text}")
        except (sr.UnknownValueError, sr.RequestError, sr.WaitTimeoutError):
            pass
            
    def draw_ui(self, frame: np.ndarray, gesture: GestureType, fps: float):
        """
        Draw user interface elements on the frame
        
        Args:
            frame: Video frame to draw on
            gesture: Current detected gesture
            fps: Current FPS
        """
        height, width = frame.shape[:2]
        
        # Draw control region
        cv2.rectangle(frame, 
                     (self.config.frame_region, self.config.frame_region),
                     (width - self.config.frame_region, height - self.config.frame_region),
                     (255, 0, 255), 2)
                     
        # Draw FPS
        if self.config.show_fps:
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), 
                       cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                       
        # Draw gesture info
        if self.config.show_gesture_info:
            gesture_text = f"Gesture: {gesture.value.upper()}"
            cv2.putText(frame, gesture_text, (20, height - 30), 
                       cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
                       
        # Draw instructions
        instructions = [
            "🤚 ARCHAND - Hand Gesture Control",
            "👆 Index: Move | 👆👆 Click | 👆👆👆👆👆 Drag",
            "👍 Thumb: Right Click | 🖐️ Fist: Scroll Up | ✋ Open: Scroll Down",
            "🖕 Middle: Speech | Press 'q' to quit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 255, 255), 1)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="🤚 ARCHAND - Advanced Hand Gesture Recognition")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    parser.add_argument("--fps", type=int, default=60, help="Camera FPS")
    parser.add_argument("--no-fps", action="store_true", help="Hide FPS display")
    parser.add_argument("--no-gesture-info", action="store_true", help="Hide gesture info")
    parser.add_argument("--sensitivity", type=int, default=8, help="Mouse sensitivity (1-20)")
    
    return parser.parse_args()


def main():
    """Main application entry point"""
    print("🤚 ARCHAND - Advanced Hand Gesture Recognition System")
    print("=" * 60)
    
    # Parse arguments
    args = parse_arguments()
    
    # Create configuration
    config = Config(
        camera_width=args.width,
        camera_height=args.height,
        camera_fps=args.fps,
        smoothing_factor=args.sensitivity,
        show_fps=not args.no_fps,
        show_gesture_info=not args.no_gesture_info
    )
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera. Please check camera permissions.")
        print("💡 Tip: Go to System Preferences > Security & Privacy > Camera")
        return 1
        
    # Configure camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)
    cap.set(cv2.CAP_PROP_FPS, config.camera_fps)
    
    # Initialize gesture recognizer
    recognizer = HandGestureRecognizer(config)
    
    print("🚀 Starting gesture recognition...")
    print("📋 Gesture Guide:")
    print("   👆 Index finger: Move mouse")
    print("   👆👆 Index + Middle: Click")
    print("   👍 Thumb: Right click")
    print("   🖐️ Fist: Scroll up")
    print("   ✋ Open palm: Scroll down")
    print("   🖕 Middle finger: Speech to text")
    print("   Press 'q' to quit")
    print("-" * 60)
    
    # Main loop
    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("❌ Error: Could not read frame from camera")
                break
                
            # Process frame
            frame = recognizer.detect_hands(frame)
            landmarks, _ = recognizer.get_hand_landmarks(frame)
            
            # Detect and execute gestures
            if len(landmarks) >= 21:  # Full hand detected
                fingers = recognizer.detect_finger_states(landmarks)
                gesture = recognizer.classify_gesture(fingers)
                recognizer.execute_gesture(gesture, landmarks, frame)
            else:
                gesture = GestureType.MOVE
                
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - recognizer.previous_time) if recognizer.previous_time > 0 else 0
            recognizer.previous_time = current_time
            
            # Draw UI
            recognizer.draw_ui(frame, gesture, fps)
            
            # Display frame
            cv2.imshow("🤚 ARCHAND - Hand Gesture Control", frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Application interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("👋 ARCHAND session ended. Thank you!")
        
    return 0


if __name__ == "__main__":
    sys.exit(main())