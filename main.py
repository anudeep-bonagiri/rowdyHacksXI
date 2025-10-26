#!/usr/bin/env python3
"""
jarvAIs - Hand Gesture & Voice Control System
============================================

A comprehensive computer control system that combines hand gesture recognition 
with voice commands for an intuitive, hands-free computing experience.

Features:
- Mouse control with hand tracking
- Multi-gesture recognition (click, drag, scroll)
- Voice control with JARVIS integration
- Auto-start voice recognition
- Real-time performance with 60 FPS
- Quick access to common applications

Usage:
- Run without arguments: Hybrid mode with auto-start
- Run with --help: Show help

Author: AI Assistant
License: MIT
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import autopy
import speech_recognition as sr
import argparse
import sys
import os
import threading
import time
import subprocess
import webbrowser
from datetime import datetime
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
    NONE = "none"


@dataclass
class Config:
    """Configuration class for the hybrid control system"""
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
    
    # Speech settings - Enhanced for better clarity
    speech_timeout: int = 5  # Shorter timeout for faster response
    speech_phrase_limit: int = 10  # Shorter phrase limit for better accuracy
    typing_interval: float = 0.01
    auto_start: bool = True
    live_transcribe: bool = True  # Enable live transcription
    
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
        """Initialize speech recognition system with enhanced settings"""
        try:
            self.speech_recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Enhanced speech recognition settings for better clarity
            self.speech_recognizer.energy_threshold = 200  # Lower threshold for better sensitivity
            self.speech_recognizer.dynamic_energy_threshold = True
            self.speech_recognizer.dynamic_energy_adjustment_damping = 0.15
            self.speech_recognizer.dynamic_energy_ratio = 1.5
            self.speech_recognizer.pause_threshold = 0.8  # Shorter pause detection
            self.speech_recognizer.phrase_threshold = 0.3  # Faster phrase detection
            self.speech_recognizer.non_speaking_duration = 0.5  # Shorter non-speaking duration
            
            # Calibrate microphone with longer duration for better noise reduction
            print("🎤 Calibrating microphone for optimal audio quality...")
            with self.microphone as source:
                self.speech_recognizer.adjust_for_ambient_noise(source, duration=2.0)
            print("✅ Speech recognition initialized successfully")
        except Exception as e:
            print(f"❌ Speech recognition unavailable: {e}")
            self.speech_recognizer = None
            self.microphone = None
            
    def setup_screen_info(self):
        """Get screen dimensions for mouse mapping"""
        self.screen_width, self.screen_height = autopy.screen.size()
        print(f"🖥️ Screen resolution: {self.screen_width}x{self.screen_height}")
        
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
        
    def get_hand_landmarks(self, frame: np.ndarray, hand_index: int = 0, draw: bool = True) -> tuple:
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
        
    def detect_finger_states(self, landmarks: list) -> list:
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
        
    def classify_gesture(self, fingers: list) -> GestureType:
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
        
    def execute_gesture(self, gesture: GestureType, landmarks: list, frame: np.ndarray):
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
        """Handle speech to text conversion with enhanced accuracy"""
        if not self.speech_recognizer or not self.microphone:
            print("❌ Speech recognition not available")
            return
            
        try:
            print("🎤 Listening... Speak clearly now!")
            with self.microphone as source:
                # Recalibrate for current environment
                self.speech_recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Listen with enhanced settings
                audio = self.speech_recognizer.listen(
                    source, 
                    timeout=self.config.speech_timeout,
                    phrase_time_limit=self.config.speech_phrase_limit
                )
                
                print("🔄 Processing speech...")
                
                # Try multiple recognition methods for better accuracy
                text = None
                
                # Primary: Google Speech Recognition
                try:
                    text = self.speech_recognizer.recognize_google(audio, language='en-US')
                    print(f"✅ Google: {text}")
                except sr.UnknownValueError:
                    print("❌ Google couldn't understand audio")
                except sr.RequestError as e:
                    print(f"❌ Google error: {e}")
                
                # Fallback: Try with different language models
                if not text:
                    try:
                        text = self.speech_recognizer.recognize_google(audio, language='en-GB')
                        print(f"✅ Google UK: {text}")
                    except:
                        pass
                
                # If we got text, process it
                if text:
                    text = text.strip().lower()
                    print(f"🎤 Speech recognized: '{text}'")
                    
                    # Process voice commands first
                    if self._process_voice_command(text):
                        return
                    
                    # If not a command, type the text
                    pyautogui.write(text, interval=self.config.typing_interval)
                    print(f"📝 Typed: {text}")
                else:
                    print("❌ Could not understand speech. Please try again.")
                    
        except sr.WaitTimeoutError:
            print("⏰ Speech timeout - no audio detected")
        except Exception as e:
            print(f"❌ Speech error: {e}")
    
    def _process_voice_command(self, text: str) -> bool:
        """Process voice commands and return True if command was processed"""
        text = text.lower().strip()
        
        # Voice commands
        if "open" in text:
            if "calculator" in text or "calc" in text:
                subprocess.Popen("calc.exe")
                print("🧮 Calculator opened")
                return True
            elif "notepad" in text or "note" in text:
                subprocess.Popen("notepad.exe")
                print("📝 Notepad opened")
                return True
            elif "browser" in text or "chrome" in text or "web" in text:
                webbrowser.open("https://www.google.com")
                print("🌐 Browser opened")
                return True
            elif "google" in text:
                webbrowser.open("https://www.google.com")
                print("🌐 Google opened")
                return True
        
        elif "volume" in text:
            if "up" in text or "increase" in text:
                pyautogui.press("volumeup")
                print("🔊 Volume increased")
                return True
            elif "down" in text or "decrease" in text:
                pyautogui.press("volumedown")
                print("🔉 Volume decreased")
                return True
            elif "mute" in text:
                pyautogui.press("volumemute")
                print("🔇 Volume muted")
                return True
        
        elif "press" in text:
            if "enter" in text:
                pyautogui.press("enter")
                print("⏎ Enter pressed")
                return True
            elif "space" in text:
                pyautogui.press("space")
                print("␣ Space pressed")
                return True
            elif "tab" in text:
                pyautogui.press("tab")
                print("⇥ Tab pressed")
                return True
            elif "escape" in text or "esc" in text:
                pyautogui.press("escape")
                print("⎋ Escape pressed")
                return True
        
        elif "close" in text or "quit" in text:
            if "application" in text or "app" in text:
                pyautogui.hotkey("alt", "f4")
                print("❌ Application closed")
                return True
            elif "window" in text:
                pyautogui.hotkey("alt", "f4")
                print("❌ Window closed")
                return True
        
        elif "screenshot" in text or "screen shot" in text:
            pyautogui.hotkey("win", "shift", "s")
            print("📸 Screenshot taken")
            return True
        
        return False
            
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
            "jarvAIs - Hand Gesture & Voice Control",
            "Index: Move | Index+Middle: Click | Drag",
            "Thumb: Right Click | Fist: Scroll Up | Open: Scroll Down",
            "Middle: Speech | Voice: 'JARVIS' + command | Press 'q' to quit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 255, 255), 1)
        
    


class JARVISVoiceController:
    """JARVIS Voice Control for jarvAIs"""
    
    def __init__(self, config: Config):
        self.config = config
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.listening = False
        self.running = True
        self.live_transcription = ""  # Store live transcription
        self.last_transcription = ""  # Store last successful transcription
        self.audio_level = 0.0  # Current audio level (0.0 to 1.0)
        self.audio_waves = []  # Store wave data for visualization
        
        # Voice commands
        self.commands = {
            "calculator": self._open_calculator,
            "calc": self._open_calculator,
            "notepad": self._open_notepad,
            "note": self._open_notepad,
            "browser": self._open_browser,
            "web": self._open_browser,
            "chrome": self._open_browser,
            "time": self._tell_time,
            "clock": self._tell_time,
            "date": self._tell_date,
            "today": self._tell_date,
            "joke": self._tell_joke,
            "funny": self._tell_joke,
            "volume up": self._volume_up,
            "louder": self._volume_up,
            "volume down": self._volume_down,
            "quieter": self._volume_down,
            "mute": self._volume_down,
            "minimize": self._minimize_window,
            "close": self._close_window,
            "exit": self._close_window,
            "quit": self._quit_jarvis,
            "stop": self._quit_jarvis,
            "shutdown": self._quit_jarvis
        }
        
        self.wake_words = ["jarvis", "hey jarvis", "okay jarvis"]
        self.setup_microphone()
        
    def setup_microphone(self):
        """Setup microphone with error handling"""
        try:
            self.microphone = sr.Microphone()
            # Test microphone access
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
            print("🎤 JARVIS microphone initialized and tested")
        except Exception as e:
            print(f"❌ JARVIS microphone error: {e}")
            self.microphone = None
            
    def start_listening(self):
        """Start voice recognition"""
        if not self.microphone:
            print("❌ JARVIS microphone not available!")
            return False
        self.listening = True
        self.listen_thread = threading.Thread(target=self._listen_continuously, daemon=True)
        self.listen_thread.start()
        print("🎤 JARVIS voice recognition started!")
        return True
        
    def stop_listening(self):
        """Stop voice recognition"""
        self.listening = False
        print("⏹️ JARVIS voice recognition stopped.")
        
    def start_auto(self):
        """Auto-start JARVIS with voice recognition"""
        if self.config.auto_start:
            print("🚀 JARVIS auto-starting...")
            self.start_listening()
    
    def get_live_transcription(self):
        """Get current live transcription"""
        return self.live_transcription
    
    def get_last_transcription(self):
        """Get last successful transcription"""
        return self.last_transcription
    
    def get_audio_level(self):
        """Get current audio level"""
        return self.audio_level
    
    def get_audio_waves(self):
        """Get current audio wave data"""
        return self.audio_waves
        
    def _listen_continuously(self):
        """Continuous listening loop with live transcription and improved error handling"""
        while self.listening and self.running and self.microphone:
            try:
                # Create new microphone instance for each listen attempt
                mic = sr.Microphone()
                with mic as source:
                    # Optimize recognizer settings for better performance
                    self.recognizer.energy_threshold = 300
                    self.recognizer.dynamic_energy_threshold = True
                    self.recognizer.pause_threshold = 1.0  # Increased from 0.5
                    self.recognizer.phrase_threshold = 0.3  # Increased from 0.2
                    self.recognizer.non_speaking_duration = 0.5  # Increased from 0.2
                    
                    # Listen with longer timeout to reduce errors
                    audio = self.recognizer.listen(source, timeout=2.0, phrase_time_limit=self.config.speech_phrase_limit)
                    
                try:
                    # Calculate audio level for visualization
                    audio_data = audio.get_raw_data()
                    audio_level = self._calculate_audio_level(audio_data)
                    self.audio_level = audio_level
                    
                    # Generate wave data for visualization
                    self._generate_wave_data(audio_level)
                    
                    text = self.recognizer.recognize_google(audio).lower()
                    
                    # Update live transcription
                    if self.config.live_transcribe:
                        self.live_transcription = text
                        self.last_transcription = text
                        print(f"🎤 Live: {text}")
                    
                    # Check for wake words
                    for wake_word in self.wake_words:
                        if wake_word in text:
                            command = text.replace(wake_word, "").strip()
                            print(f"🤖 JARVIS heard: {text}")
                            self._execute_command(command)
                            break
                    else:
                        # If no wake word found but we have text, show it
                        if text and not any(wake_word in text for wake_word in self.wake_words):
                            print(f"💬 Heard: {text} (no wake word)")
                            
                except sr.UnknownValueError:
                    # Could not understand audio, but still show audio level
                    audio_data = audio.get_raw_data()
                    audio_level = self._calculate_audio_level(audio_data)
                    self.audio_level = audio_level
                    self._generate_wave_data(audio_level)
                    
                    if self.config.live_transcribe:
                        print("🔇 Could not understand audio...")
                    continue
                except sr.RequestError as e:
                    print(f"❌ JARVIS recognition error: {e}")
                    time.sleep(2)  # Wait longer before retrying
                    continue
                except sr.WaitTimeoutError:
                    # No speech detected, continue listening
                    if self.config.live_transcribe:
                        print("⏳ Listening...")
                    continue
                    
            except Exception as e:
                print(f"❌ JARVIS listening error: {e}")
                time.sleep(1.0)  # Longer wait time
                
    def _execute_command(self, command):
        """Execute voice command"""
        if not command:
            return
            
        print(f"🤖 JARVIS executing: '{command}'")
        
        # Try exact match first
        if command in self.commands:
            try:
                self.commands[command]()
                print(f"✅ JARVIS executed: {command}")
                return
            except Exception as e:
                print(f"❌ JARVIS error: {e}")
                return
        
        # Try partial matches
        for cmd_key, cmd_func in self.commands.items():
            if cmd_key in command or command in cmd_key:
                try:
                    cmd_func()
                    print(f"✅ JARVIS executed: {cmd_key}")
                    return
                except Exception as e:
                    print(f"❌ JARVIS error: {e}")
                    return
        
        # No command found
        print(f"❓ JARVIS: Unknown command '{command}'")
            
    def _open_calculator(self):
        """Open calculator"""
        subprocess.Popen("calc.exe")
        print("🧮 Calculator opened")
        
    def _open_notepad(self):
        """Open notepad"""
        subprocess.Popen("notepad.exe")
        print("📝 Notepad opened")
        
    def _open_browser(self):
        """Open web browser"""
        webbrowser.open("https://www.google.com")
        print("🌐 Browser opened")
        
    def _tell_time(self):
        """Tell current time"""
        current_time = datetime.now().strftime("%I:%M %p")
        self._speak(f"The time is {current_time}")
        
    def _tell_date(self):
        """Tell current date"""
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        self._speak(f"Today is {current_date}")
        
    def _tell_joke(self):
        """Tell a joke"""
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "Why did the scarecrow win an award? He was outstanding in his field!",
            "Why don't eggs tell jokes? They'd crack each other up!",
            "What do you call a fake noodle? An impasta!",
            "Why did the math book look so sad? Because it had too many problems!"
        ]
        import random
        joke = random.choice(jokes)
        self._speak(joke)
        
    def _volume_up(self):
        """Increase volume"""
        import pyautogui
        pyautogui.press('volumeup')
        print("🔊 Volume increased")
        
    def _volume_down(self):
        """Decrease volume"""
        import pyautogui
        pyautogui.press('volumedown')
        print("🔉 Volume decreased")
        
    def _minimize_window(self):
        """Minimize current window"""
        import pyautogui
        pyautogui.hotkey('alt', 'f9')
        print("📱 Window minimized")
        
    def _close_window(self):
        """Close current window"""
        import pyautogui
        pyautogui.hotkey('alt', 'f4')
        print("❌ Window closed")
        
    def _quit_jarvis(self):
        """Quit JARVIS"""
        print("👋 JARVIS shutting down...")
        self.running = False
        self.listening = False
        sys.exit(0)
        
    def _calculate_audio_level(self, audio_data):
        """Calculate intensive audio level from raw audio data with enhanced sensitivity"""
        try:
            import struct
            import math
            
            # Convert bytes to 16-bit integers
            audio_samples = struct.unpack('<' + 'h' * (len(audio_data) // 2), audio_data)
            
            # Calculate multiple audio metrics for more sensitivity
            rms = (sum(sample * sample for sample in audio_samples) / len(audio_samples)) ** 0.5
            
            # Calculate peak level for more dramatic response
            peak = max(abs(sample) for sample in audio_samples)
            
            # Calculate spectral centroid for frequency content
            fft_samples = audio_samples[:min(1024, len(audio_samples))]  # Limit for performance
            spectral_centroid = sum(abs(s) for s in fft_samples) / len(fft_samples)
            
            # Combine multiple metrics for more intensive response
            rms_level = min(rms / 16384.0, 1.0)  # More sensitive threshold
            peak_level = min(peak / 16384.0, 1.0)  # More sensitive threshold
            spectral_level = min(spectral_centroid / 8192.0, 1.0)  # Frequency content
            
            # Weighted combination for more dramatic effect
            combined_level = (rms_level * 0.4 + peak_level * 0.4 + spectral_level * 0.2)
            
            # Apply exponential scaling for more dramatic response
            intensive_level = combined_level ** 0.7  # Makes lower levels more visible
            
            # Add some persistence for smoother visualization
            if hasattr(self, 'last_audio_level'):
                intensive_level = intensive_level * 0.7 + self.last_audio_level * 0.3
            
            self.last_audio_level = intensive_level
            return min(intensive_level, 1.0)
            
        except:
            return 0.0
    
    def _generate_wave_data(self, audio_level):
        """Generate audio meter/equalizer bar data like professional audio software"""
        import time
        import math
        
        current_time = time.time()
        
        # Create audio meter bars (like VU meter or equalizer)
        meter_bars = []
        num_bars = 20  # Number of frequency bars
        
        # Generate frequency bars with different responses
        for i in range(num_bars):
            # Different frequency ranges for each bar
            freq_factor = i / num_bars  # 0.0 to 1.0
            
            # Simulate different frequency responses
            if freq_factor < 0.3:  # Low frequencies (bass)
                bar_level = audio_level * (0.8 + 0.4 * math.sin(current_time * 2))
            elif freq_factor < 0.7:  # Mid frequencies (vocals)
                bar_level = audio_level * (1.0 + 0.6 * math.sin(current_time * 4))
            else:  # High frequencies (treble)
                bar_level = audio_level * (0.6 + 0.8 * math.sin(current_time * 6))
            
            # Add some randomness for realistic meter behavior
            noise = (hash(str(current_time + i)) % 100) / 1000.0  # Small random variation
            bar_level = max(0, min(1.0, bar_level + noise))
            
            # Add decay for realistic meter behavior
            if hasattr(self, 'last_bar_levels') and i < len(self.last_bar_levels):
                decay_factor = 0.85  # How fast bars fall
                bar_level = max(bar_level, self.last_bar_levels[i] * decay_factor)
            
            meter_bars.append({
                'bar_index': i,
                'level': bar_level,
                'height': int(bar_level * 100),  # Height in pixels
                'time': current_time,
                'frequency_band': freq_factor
            })
        
        # Store current levels for decay calculation
        self.last_bar_levels = [bar['level'] for bar in meter_bars]
        
        # Keep meter data for visualization
        self.audio_waves = meter_bars
    
    def _speak(self, text):
        """Text to speech"""
        try:
            os.system(f'powershell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{text}\')"')
        except Exception as e:
            print(f"JARVIS speech error: {e}")


def main():
    """Main application entry point - Hybrid Hand Gesture & Voice Control Mode"""
    print("🤖 jarvAIs - Hand Gesture & Voice Control System")
    print("=" * 60)
    print("👋 Hand Gestures:")
    print("   Index finger: Move mouse")
    print("   Index + Middle: Click")
    print("   Thumb: Right click")
    print("   Fist: Scroll up")
    print("   Open palm: Scroll down")
    print("   Middle finger: Speech to text")
    print("")
    print("🎤 Voice Commands:")
    print("   Say 'JARVIS' followed by: calculator, notepad, browser, time, date, joke, volume up/down, minimize, close, quit")
    print("")
    print("📝 Live Transcription: ENABLED")
    print("   - Shows what JARVIS hears in real-time")
    print("   - Commands with 'JARVIS' wake word are executed")
    print("   - Other speech is transcribed for reference")
    print("=" * 60)
    
    # Create configuration
    config = Config()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera. Please check camera permissions.")
        return 1
        
    # Configure camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)
    cap.set(cv2.CAP_PROP_FPS, config.camera_fps)
    
    # Initialize JARVIS voice control first
    jarvis = JARVISVoiceController(config)
    
    # Initialize gesture recognizer with jarvis reference
    recognizer = HandGestureRecognizer(config)
    recognizer.jarvis = jarvis  # Pass jarvis instance for live transcription
    
    # Auto-start voice recognition
    jarvis.start_auto()
    
    print("🚀 jarvAIs started! Press 'q' to quit.")
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("❌ Error: Could not read frame from camera")
                break
                
            # Process frame for hand gestures
            frame = recognizer.detect_hands(frame)
            landmarks, _ = recognizer.get_hand_landmarks(frame)
            
            # Detect and execute gestures
            if len(landmarks) >= 21:  # Full hand detected
                fingers = recognizer.detect_finger_states(landmarks)
                gesture = recognizer.classify_gesture(fingers)
                recognizer.execute_gesture(gesture, landmarks, frame)
            else:
                gesture = GestureType.NONE
                
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - recognizer.previous_time) if recognizer.previous_time > 0 else 0
            recognizer.previous_time = current_time
            
            # Draw UI
            recognizer.draw_ui(frame, gesture, fps)
            
            # Display frame
            cv2.imshow("jarvAIs - Hand Gesture & Voice Control", frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n👋 jarvAIs shutting down...")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        jarvis.stop_listening()
        print("✅ jarvAIs session ended. Thank you!")
        
    return 0


if __name__ == "__main__":
    sys.exit(main())