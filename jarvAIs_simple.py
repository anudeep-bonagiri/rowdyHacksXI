#!/usr/bin/env python3
"""
jarvAIs - Simplified Hand Gesture & Voice Control System
======================================================

A streamlined computer control system that combines hand gesture recognition 
with voice commands for hands-free computing.

Features:
- Mouse control with hand tracking
- Multi-gesture recognition (click, drag, scroll, YouTube)
- Voice control with JARVIS integration
- Real-time performance
- No GUI - pure functionality

Author: AI Assistant
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
import os
import threading
import subprocess
import webbrowser
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
    YOUTUBE = "youtube"
    NONE = "none"


@dataclass
class Config:
    """Configuration class for the gesture recognition system"""
    # Camera settings
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    
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


class HandGestureRecognizer:
    """
    Advanced hand gesture recognition system using MediaPipe
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Screen dimensions
        self.screen_width, self.screen_height = autopy.screen.size()
        print(f"Screen resolution: {self.screen_width}x{self.screen_height}")
        
        # Mouse control variables
        self.previous_x = 0
        self.previous_y = 0
        self.current_x = 0
        self.current_y = 0
        self.previous_time = 0
        self.gesture_history = []  # Track recent gestures for stability
        self.last_mouse_update = 0  # Track last mouse update time
        self.mouse_update_interval = 50  # Update mouse every 50ms
        self.stable_gesture_count = 0  # Count consecutive stable gestures
        self.youtube_cooldown = False  # YouTube gesture cooldown
        self.stability_buffer = []
        self.drag_hold = False
        
        # Disable pyautogui failsafe
        pyautogui.FAILSAFE = False
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Calibrate microphone
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
        print("JARVIS microphone initialized and tested")
        
        # Test speech recognition
        try:
            with self.microphone as source:
                self.recognizer.listen(source, timeout=1)
            print("Speech recognition initialized successfully")
        except sr.WaitTimeoutError:
            print("Speech recognition initialized (no initial audio detected)")
    
    def is_gesture_stable(self, current_gesture: GestureType) -> bool:
        """
        Check if the current gesture is stable by looking at recent gesture history
        
        Args:
            current_gesture: The current detected gesture
            
        Returns:
            True if gesture is stable, False otherwise
        """
        # Add current gesture to history
        self.gesture_history.append(current_gesture)
        
        # Keep only last 5 gestures
        if len(self.gesture_history) > 5:
            self.gesture_history.pop(0)
            
        # Check if we have enough history
        if len(self.gesture_history) < 3:
            return False
            
        # Check if last 3 gestures are the same
        last_three = self.gesture_history[-3:]
        return all(gesture == current_gesture for gesture in last_three)
    
    def classify_gesture(self, landmarks: List) -> GestureType:
        """
        Classify hand gesture based on finger positions
        
        Args:
            landmarks: List of hand landmarks
            
        Returns:
            GestureType enum value
        """
        if len(landmarks) < 21:
            return GestureType.NONE
            
        # Get finger tip positions
        fingers = []
        
        # Thumb (landmark 4)
        if landmarks[4][1] > landmarks[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)
            
        # Index finger (landmark 8)
        if landmarks[8][2] < landmarks[6][2]:
            fingers.append(1)
        else:
            fingers.append(0)
            
        # Middle finger (landmark 12)
        if landmarks[12][2] < landmarks[10][2]:
            fingers.append(1)
        else:
            fingers.append(0)
            
        # Ring finger (landmark 16)
        if landmarks[16][2] < landmarks[14][2]:
            fingers.append(1)
        else:
            fingers.append(0)
            
        # Pinky (landmark 20)
        if landmarks[20][2] < landmarks[18][2]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Gesture classification
        if fingers == [0, 1, 0, 0, 0]:  # Only index finger up
            return GestureType.MOVE
        elif fingers == [0, 1, 1, 0, 0]:  # Index and middle finger up
            return GestureType.CLICK
        elif fingers == [0, 1, 1, 1, 0]:  # Index, middle, ring finger up
            return GestureType.DOUBLE_CLICK
        elif fingers == [1, 0, 0, 0, 0]:  # Only thumb up
            return GestureType.RIGHT_CLICK
        elif fingers == [0, 1, 1, 1, 1]:  # All fingers except thumb up
            return GestureType.SCROLL_UP
        elif fingers == [0, 0, 0, 0, 0]:  # All fingers down
            return GestureType.SCROLL_DOWN
        elif fingers == [0, 0, 0, 0, 1]:  # Only pinky up
            return GestureType.SPEECH
        elif fingers == [0, 1, 1, 0, 1]:  # Index, middle, pinky up
            return GestureType.DRAG
        # YouTube (thumb + index touching)
        elif fingers == [1, 1, 0, 0, 0]:
            return GestureType.YOUTUBE
        else:
            return GestureType.NONE
    
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
            
        # Check gesture stability
        if not self.is_gesture_stable(gesture):
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
            # Only update mouse if enough time has passed and movement is significant
            current_time = time.time() * 1000  # Convert to milliseconds
            movement_threshold = 5  # Minimum pixels to move
            
            # Calculate movement distance
            movement_distance = ((self.current_x - self.previous_x) ** 2 + 
                               (self.current_y - self.previous_y) ** 2) ** 0.5
            
            if (current_time - self.last_mouse_update > self.mouse_update_interval and 
                movement_distance > movement_threshold):
                self._handle_mouse_move()
                self.last_mouse_update = current_time
            self._stop_drag()  # Stop drag if not in drag gesture
            
        elif gesture == GestureType.CLICK:
            self._handle_click(x, y, frame)
            self._stop_drag()
            
        elif gesture == GestureType.DOUBLE_CLICK:
            self._handle_double_click(x, y, frame)
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
            self._handle_speech()
            self._stop_drag()
            
        elif gesture == GestureType.YOUTUBE:
            self._handle_youtube()
            self._stop_drag()
        
        # Update previous position
        self.previous_x = self.current_x
        self.previous_y = self.current_y
    
    def _handle_mouse_move(self):
        """Handle mouse movement"""
        try:
            autopy.mouse.move(self.current_x, self.current_y)
        except Exception as e:
            print(f"Mouse move error: {e}")
    
    def _handle_click(self, x: int, y: int, frame: np.ndarray):
        """Handle left click"""
        try:
            autopy.mouse.click()
            print("Left click executed")
        except Exception as e:
            print(f"Click error: {e}")
    
    def _handle_double_click(self, x: int, y: int, frame: np.ndarray):
        """Handle double click"""
        try:
            autopy.mouse.click()
            time.sleep(0.1)
            autopy.mouse.click()
            print("Double click executed")
        except Exception as e:
            print(f"Double click error: {e}")
    
    def _handle_right_click(self, x: int, y: int, frame: np.ndarray):
        """Handle right click"""
        try:
            autopy.mouse.click(autopy.mouse.Button.RIGHT)
            print("Right click executed")
        except Exception as e:
            print(f"Right click error: {e}")
    
    def _handle_drag(self):
        """Handle drag operation"""
        if not self.drag_hold:
            try:
                autopy.mouse.toggle(True)
                self.drag_hold = True
                print("Drag started")
            except Exception as e:
                print(f"Drag start error: {e}")
    
    def _stop_drag(self):
        """Stop drag operation"""
        if self.drag_hold:
            try:
                autopy.mouse.toggle(False)
                self.drag_hold = False
                print("Drag stopped")
            except Exception as e:
                print(f"Drag stop error: {e}")
    
    def _handle_scroll_up(self):
        """Handle scroll up"""
        try:
            pyautogui.scroll(self.config.scroll_up_speed)
            print("Scroll up executed")
        except Exception as e:
            print(f"Scroll up error: {e}")
    
    def _handle_scroll_down(self):
        """Handle scroll down"""
        try:
            pyautogui.scroll(self.config.scroll_down_speed)
            print("Scroll down executed")
        except Exception as e:
            print(f"Scroll down error: {e}")
    
    def _handle_speech(self):
        """Handle speech recognition"""
        try:
            print("Listening for voice command...")
            with self.microphone as source:
                # Listen for audio with timeout
                audio = self.recognizer.listen(source, timeout=self.config.speech_timeout, 
                                             phrase_time_limit=self.config.speech_phrase_limit)
            
            # Recognize speech
            text = self.recognizer.recognize_google(audio)
            print(f"Voice command: {text}")
            
            # Process voice commands
            self._process_voice_command(text.lower())
            
        except sr.WaitTimeoutError:
            print("Voice command timeout")
        except sr.UnknownValueError:
            print("Could not understand voice command")
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
        except Exception as e:
            print(f"Speech handling error: {e}")
    
    def _process_voice_command(self, command: str):
        """Process voice commands"""
        try:
            if "open" in command:
                if "chrome" in command:
                    subprocess.run(["open", "-a", "Google Chrome"])
                    print("Opening Chrome...")
                elif "safari" in command:
                    subprocess.run(["open", "-a", "Safari"])
                    print("Opening Safari...")
                elif "finder" in command:
                    subprocess.run(["open", "-a", "Finder"])
                    print("Opening Finder...")
                elif "terminal" in command:
                    subprocess.run(["open", "-a", "Terminal"])
                    print("Opening Terminal...")
                elif "youtube" in command:
                    webbrowser.open("https://www.youtube.com/")
                    print("Opening YouTube...")
                elif "google" in command:
                    webbrowser.open("https://www.google.com/")
                    print("Opening Google...")
                else:
                    print(f"Unknown application: {command}")
            
            elif "volume" in command:
                if "up" in command or "increase" in command:
                    pyautogui.press("volumeup")
                    print("Volume increased")
                elif "down" in command or "decrease" in command:
                    pyautogui.press("volumedown")
                    print("Volume decreased")
                elif "mute" in command:
                    pyautogui.press("volumemute")
                    print("Volume muted")
            
            elif "type" in command:
                # Extract text to type
                text_to_type = command.replace("type", "").strip()
                if text_to_type:
                    pyautogui.typewrite(text_to_type, interval=self.config.typing_interval)
                    print(f"Typed: {text_to_type}")
            
            elif "press" in command:
                if "enter" in command:
                    pyautogui.press("enter")
                    print("Pressed Enter")
                elif "space" in command:
                    pyautogui.press("space")
                    print("Pressed Space")
                elif "tab" in command:
                    pyautogui.press("tab")
                    print("Pressed Tab")
                elif "escape" in command or "esc" in command:
                    pyautogui.press("escape")
                    print("Pressed Escape")
            
            elif "close" in command or "quit" in command:
                if "application" in command or "app" in command:
                    pyautogui.hotkey("cmd", "q")
                    print("Closing application")
                elif "window" in command:
                    pyautogui.hotkey("cmd", "w")
                    print("Closing window")
            
            elif "screenshot" in command:
                pyautogui.hotkey("cmd", "shift", "3")
                print("Taking screenshot")
            
            else:
                print(f"Unknown voice command: {command}")
                
        except Exception as e:
            print(f"Voice command processing error: {e}")
    
    def _handle_youtube(self):
        """Handle YouTube opening gesture - opens Chrome and navigates to YouTube"""
        if not self.youtube_cooldown:
            try:
                # Try to open Chrome specifically first
                chrome_paths = [
                    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
                    '/usr/bin/google-chrome',
                    '/usr/bin/chromium-browser',
                    'chrome',
                    'google-chrome'
                ]
                
                chrome_opened = False
                for chrome_path in chrome_paths:
                    try:
                        # Open Chrome with YouTube URL
                        subprocess.run([chrome_path, 'https://www.youtube.com/'], 
                                     check=True, timeout=5)
                        print("Opening YouTube in Chrome...")
                        chrome_opened = True
                        break
                    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                        continue
                
                # Fallback to default browser if Chrome not found
                if not chrome_opened:
                    webbrowser.open("https://www.youtube.com/")
                    print("Opening YouTube in default browser...")
                
                # Set cooldown to prevent multiple opens
                self.youtube_cooldown = True
                threading.Timer(3.0, lambda: setattr(self, 'youtube_cooldown', False)).start()
                
            except Exception as e:
                print(f"YouTube opening error: {e}")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, GestureType]:
        """
        Process a single frame for gesture recognition
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (processed_frame, detected_gesture)
        """
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hands
        results = self.hands.process(rgb_frame)
        
        gesture = GestureType.NONE
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Convert landmarks to list format
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append([
                        landmark.x,
                        landmark.x * self.config.camera_width,
                        landmark.y * self.config.camera_height
                    ])
                
                # Classify gesture
                gesture = self.classify_gesture(landmarks)
                
                # Execute gesture
                if gesture != GestureType.NONE:
                    self.execute_gesture(gesture, landmarks, frame)
                
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        return frame, gesture


def main():
    """Main function to run the gesture recognition system"""
    parser = argparse.ArgumentParser(description='jarvAIs - Hand Gesture & Voice Control System')
    parser.add_argument('--fps', type=int, default=30, help='Camera FPS (default: 30)')
    parser.add_argument('--width', type=int, default=640, help='Camera width (default: 640)')
    parser.add_argument('--height', type=int, default=480, help='Camera height (default: 480)')
    parser.add_argument('--smoothing', type=int, default=8, help='Mouse smoothing factor (default: 8)')
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config(
        camera_width=args.width,
        camera_height=args.height,
        camera_fps=args.fps,
        smoothing_factor=args.smoothing
    )
    
    # Initialize gesture recognizer
    recognizer = HandGestureRecognizer(config)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)
    cap.set(cv2.CAP_PROP_FPS, config.camera_fps)
    
    print("🚀 jarvAIs Simple Mode Started!")
    print("📋 Available Gestures:")
    print("  👆 One finger up: Move mouse")
    print("  👆👆 Two fingers up: Left click")
    print("  👆👆👆 Three fingers up: Double click")
    print("  👍 Thumb up: Right click")
    print("  ✊ All fingers up: Scroll up")
    print("  ✋ All fingers down: Scroll down")
    print("  🖕 Pinky up: Voice command")
    print("  ✋ Index+Middle+Pinky: Drag")
    print("  📺 Thumb+Index: Open YouTube")
    print("\n🎤 Voice Commands:")
    print("  'Open Chrome/Safari/Finder/Terminal'")
    print("  'Open YouTube/Google'")
    print("  'Volume up/down/mute'")
    print("  'Type [text]'")
    print("  'Press enter/space/tab/escape'")
    print("  'Close application/window'")
    print("  'Screenshot'")
    print("\nPress 'q' to quit, 'h' for help")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read camera frame")
                break
            
            # Process frame
            processed_frame, gesture = recognizer.process_frame(frame)
            
            # Display frame with gesture info
            gesture_text = f"Gesture: {gesture.value.upper()}"
            cv2.putText(processed_frame, gesture_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('jarvAIs - Hand Gesture Control', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                print("\n📋 Gesture Help:")
                print("  👆 One finger up: Move mouse")
                print("  👆👆 Two fingers up: Left click")
                print("  👆👆👆 Three fingers up: Double click")
                print("  👍 Thumb up: Right click")
                print("  ✊ All fingers up: Scroll up")
                print("  ✋ All fingers down: Scroll down")
                print("  🖕 Pinky up: Voice command")
                print("  ✋ Index+Middle+Pinky: Drag")
                print("  📺 Thumb+Index: Open YouTube")
    
    except KeyboardInterrupt:
        print("\n👋 Shutting down jarvAIs...")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("✅ jarvAIs Simple Mode Stopped")


if __name__ == "__main__":
    main()
