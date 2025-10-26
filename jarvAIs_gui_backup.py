#!/usr/bin/env python3
"""
jarvAIs - Complete Hand Gesture & Voice Control System
=====================================================

A comprehensive computer control system that combines hand gesture recognition 
with voice commands for an intuitive, hands-free computing experience.

Features:
- Mouse control with hand tracking
- Multi-gesture recognition (click, drag, scroll)
- Voice control with JARVIS integration
- GUI interface with live camera feed
- Real-time performance with 60 FPS
- Quick access to common applications
- Beautiful visual feedback

Usage:
- Run without arguments: GUI mode
- Run with --cli: Command-line mode
- Run with --help: Show help

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
import queue
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# GUI imports (only when needed)
try:
    import tkinter as tk
    from tkinter import ttk, Canvas
    from PIL import Image, ImageTk
    import subprocess
    import webbrowser
    from datetime import datetime
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


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
        self.gesture_history = []  # Track recent gestures for stability
        self.last_mouse_update = 0  # Track last mouse update time
        self.mouse_update_interval = 50  # Update mouse every 50ms
        self.stable_gesture_count = 0  # Count consecutive stable gestures
        self.youtube_cooldown = False  # YouTube gesture cooldown
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
            print("Speech recognition initialized successfully")
        except Exception as e:
            print(f"Speech recognition unavailable: {e}")
            self.speech_recognizer = None
            self.microphone = None
            self.youtube_cooldown = False
            self.speech_cooldown = False
            
    def setup_screen_info(self):
        """Get screen dimensions for mouse mapping"""
        self.screen_width, self.screen_height = autopy.screen.size()
        print(f"Screen resolution: {self.screen_width}x{self.screen_height}")
        
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
        
    def classify_gesture(self, fingers: List[int]) -> GestureType:
        """
        Classify gesture based on finger states
        
        Args:
            fingers: List of finger states
            
        Returns:
            Detected gesture type
        """
        # YouTube (thumb + index touching)
        if fingers == [1, 1, 0, 0, 0]:
            return GestureType.YOUTUBE
            
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
            
        elif gesture == GestureType.YOUTUBE:
            self._handle_youtube()
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
        
    def _handle_youtube(self):
        """Handle YouTube opening gesture - opens Chrome and navigates to YouTube"""
        if not self.youtube_cooldown:
            try:
                import subprocess
                import webbrowser
                
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
                self.root.after(3000, lambda: setattr(self, 'youtube_cooldown', False))
                
            except Exception as e:
                print(f"YouTube opening error: {e}")
                
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
                    print(f"Speech recognized: {text}")
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
            "jarvAIs - Hand Gesture Control",
            "Index: Move | Index+Middle: Click | Drag",
            "Thumb: Right Click | Fist: Scroll Up | Open: Scroll Down",
            "Middle: Speech | Press 'q' to quit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 255, 255), 1)


class JARVISVoiceController:
    """JARVIS Voice Control for jarvAIs"""
    
    def __init__(self, gui_callback=None):
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.listening = False
        self.gui_callback = gui_callback
        
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
            "exit": self._close_window
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
            print("JARVIS microphone initialized and tested")
        except Exception as e:
            print(f"JARVIS microphone error: {e}")
            self.microphone = None
            
    def start_listening(self):
        """Start voice recognition"""
        if not self.microphone:
            return False
        self.listening = True
        self.listen_thread = threading.Thread(target=self._listen_continuously, daemon=True)
        self.listen_thread.start()
        return True
        
    def stop_listening(self):
        """Stop voice recognition"""
        self.listening = False
        
    def _listen_continuously(self):
        """Continuous listening loop with improved error handling"""
        while self.listening and self.microphone:
            try:
                # Create new microphone instance for each listen attempt
                mic = sr.Microphone()
                with mic as source:
                    # Optimize recognizer settings for better performance
                    self.recognizer.energy_threshold = 300
                    self.recognizer.dynamic_energy_threshold = True
                    self.recognizer.pause_threshold = 0.5
                    self.recognizer.phrase_threshold = 0.2
                    self.recognizer.non_speaking_duration = 0.2
                    
                    # Listen with shorter timeout to avoid hanging
                    audio = self.recognizer.listen(source, timeout=0.5, phrase_time_limit=2)
                    
                try:
                    text = self.recognizer.recognize_google(audio).lower()
                    print(f"JARVIS heard: {text}")
                    
                    if self.gui_callback:
                        self.gui_callback(f"JARVIS: {text}")
                    
                    # Check for wake words
                    for wake_word in self.wake_words:
                        if wake_word in text:
                            command = text.replace(wake_word, "").strip()
                            self._execute_command(command)
                            break
                            
                except sr.UnknownValueError:
                    # Could not understand audio, continue listening
                    continue
                except sr.RequestError as e:
                    print(f"JARVIS recognition error: {e}")
                    time.sleep(1)  # Wait before retrying
                    continue
                except sr.WaitTimeoutError:
                    # No speech detected, continue listening
                    continue
                    
            except Exception as e:
                print(f"JARVIS listening error: {e}")
                time.sleep(0.5)  # Shorter wait time
                
    def _execute_command(self, command):
        """Execute voice command"""
        if not command:
            return
            
        print(f"JARVIS executing: '{command}'")
        
        # Try exact match first
        if command in self.commands:
            try:
                self.commands[command]()
                if self.gui_callback:
                    self.gui_callback(f"JARVIS executed: {command}")
                return
            except Exception as e:
                if self.gui_callback:
                    self.gui_callback(f"JARVIS error: {e}")
                return
        
        # Try partial matches
        for cmd_key, cmd_func in self.commands.items():
            if cmd_key in command or command in cmd_key:
                try:
                    cmd_func()
                    if self.gui_callback:
                        self.gui_callback(f"JARVIS executed: {cmd_key}")
                    return
                except Exception as e:
                    if self.gui_callback:
                        self.gui_callback(f"JARVIS error: {e}")
                    return
        
        # No command found
        if self.gui_callback:
            self.gui_callback(f"JARVIS: Unknown command '{command}'")
            
    def _open_calculator(self):
        """Open calculator"""
        subprocess.Popen("calc.exe")
        
    def _open_notepad(self):
        """Open notepad"""
        subprocess.Popen("notepad.exe")
        
    def _open_browser(self):
        """Open web browser"""
        webbrowser.open("https://www.google.com")
        
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
        pyautogui.press('volumeup')
        
    def _volume_down(self):
        """Decrease volume"""
        pyautogui.press('volumedown')
        
    def _minimize_window(self):
        """Minimize current window"""
        pyautogui.hotkey('alt', 'f9')
        
    def _close_window(self):
        """Close current window"""
        pyautogui.hotkey('alt', 'f4')
        
    def _speak(self, text):
        """Text to speech"""
        try:
            os.system(f'powershell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{text}\')"')
        except Exception as e:
            print(f"JARVIS speech error: {e}")


class jarvAIsGUI:
    """🚀 jarvAIs Space Cowboy Command Center - Cosmic Frontier Theme"""
    
    def __init__(self):
        if not GUI_AVAILABLE:
            print("GUI libraries not available. Please install tkinter and PIL.")
            sys.exit(1)
            
        self.root = tk.Tk()
        self.root.title("🚀 jarvAIs Space Cowboy Command Center")
        self.root.geometry("1600x1000")
        
        # Space Cowboy Theme Colors
        self.colors = {
            'deep_space': '#0a0a1a',
            'cosmic_blue': '#1a237e', 
            'neon_cyan': '#00e5ff',
            'star_gold': '#ffd700',
            'dark_bg': '#000011',
            'card_bg': '#1a1a2e',
            'accent': '#ff6b35',
            'plasma_purple': '#8e24aa'
        }
        
        # Load gesture reference images
        self.load_gesture_images()
        
        self.root.configure(bg=self.colors['deep_space'])
        
        # Make window fullscreen-capable
        self.root.state('zoomed')  # Maximize on Windows
        
        # Configuration
        self.config = Config()
        
        # Camera and processing
        self.cap = None
        self.recognizer = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=3)
        
        # JARVIS voice control
        self.jarvis = JARVISVoiceController(self._jarvis_callback)
        
        # UI state
        self.current_page = 0
        self.pages = ["main", "gestures", "voice", "tools", "settings"]
        
        # GUI elements
        self.create_widgets()
        
        # Start camera automatically
        self.root.after(1000, self.start_camera)
        
    def load_gesture_images(self):
        """Load gesture reference images from the Images folder"""
        self.gesture_images = {}
        image_files = {
            'move': 'move pointer.jpg',
            'click': 'single left click.jpg',
            'double_click': 'double left click.jpg',
            'right_click': 'single right click.jpg',
            'drag': 'Hold left click and move pointer.jpg',
            'scroll_up': 'Scroll up.jpg',
            'scroll_down': 'Scroll down.jpg',
            'speech': 'Speech to text.jpg',
            'youtube': 'Youtube.png'
        }
        
        for gesture, filename in image_files.items():
            try:
                image_path = f"Images/{filename}"
                if os.path.exists(image_path):
                    # Load and resize image for UI
                    img = Image.open(image_path)
                    img = img.resize((120, 80), Image.Resampling.LANCZOS)
                    self.gesture_images[gesture] = ImageTk.PhotoImage(img)
                    print(f"Loaded gesture image: {gesture}")
                else:
                    print(f"Image not found: {image_path}")
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
        
    def create_widgets(self):
        """Create the Space Cowboy cosmic-themed GUI interface"""
        # Create cosmic starfield background first
        self.create_cosmic_starfield()
        
        # Main container with Space Cowboy background - place it above starfield
        main_frame = tk.Frame(self.root, bg=self.colors['deep_space'])
        main_frame.place(x=0, y=0, relwidth=1, relheight=1)
        
        # 1️⃣ HEADER SECTION - Space Cowboy Command Center
        self.create_header_section(main_frame)
        
        # 2️⃣ MAIN CONTENT AREA - Horizontal layout
        self.content_frame = tk.Frame(main_frame, bg=self.colors['deep_space'])
        self.content_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Create pages (only Main and Settings)
        self.create_main_page()
        self.create_settings_page()
        
        # 3️⃣ FOOTER / NAVIGATION BAR
        self.create_footer_navigation(main_frame)
        
        # Show initial page
        self.show_page(0)
        
    def create_cosmic_starfield(self):
        """Create animated cosmic starfield background for Space Cowboy theme"""
        # Create a frame for the starfield that will be behind everything
        self.starfield_frame = tk.Frame(self.root, bg=self.colors['dark_bg'])
        self.starfield_frame.place(x=0, y=0, relwidth=1, relheight=1)
        
        self.starfield_canvas = Canvas(self.starfield_frame, bg=self.colors['dark_bg'], highlightthickness=0)
        self.starfield_canvas.pack(fill='both', expand=True)
        
        # Create cosmic stars with Space Cowboy colors
        self.stars = []
        for _ in range(200):  # More stars for cosmic space
            x = self.root.winfo_screenwidth() * 0.9 * (0.05 + 0.9 * (hash(str(_)) % 1000) / 1000)
            y = self.root.winfo_screenheight() * 0.9 * (0.05 + 0.9 * (hash(str(_ + 1000)) % 1000) / 1000)
            size = 1 + (hash(str(_ + 2000)) % 5)
            self.stars.append({
                'x': x, 'y': y, 'size': size,
                'brightness': 0.3 + 0.7 * (hash(str(_ + 3000)) % 1000) / 1000,
                'color': 'neon_cyan' if hash(str(_ + 4000)) % 8 == 0 else 
                        'star_gold' if hash(str(_ + 4000)) % 12 == 0 else 
                        'plasma_purple' if hash(str(_ + 4000)) % 15 == 0 else 'white'
            })
        
        self.animate_cosmic_stars()
        
    def animate_cosmic_stars(self):
        """Animate the cosmic starfield with Space Cowboy pulsing"""
        self.starfield_canvas.delete("all")
        
        for star in self.stars:
            # Enhanced pulsing effect for cosmic space
            brightness = star['brightness'] + 0.5 * (hash(str(star['x'] + star['y'])) % 1000) / 1000
            brightness = min(1.0, brightness)
            
            # Space Cowboy star colors
            if star['color'] == 'neon_cyan':
                color = f"#{int(0 * brightness):02x}{int(229 * brightness):02x}{int(255 * brightness):02x}"
            elif star['color'] == 'star_gold':
                color = f"#{int(255 * brightness):02x}{int(215 * brightness):02x}{int(0 * brightness):02x}"
            elif star['color'] == 'plasma_purple':
                color = f"#{int(142 * brightness):02x}{int(36 * brightness):02x}{int(170 * brightness):02x}"
            else:
                color = f"#{int(255 * brightness):02x}{int(255 * brightness):02x}{int(255 * brightness):02x}"
            
            self.starfield_canvas.create_oval(
                star['x'] - star['size'], star['y'] - star['size'],
                star['x'] + star['size'], star['y'] + star['size'],
                fill=color, outline=color
            )
        
        # Schedule next animation
        self.root.after(60, self.animate_cosmic_stars)
        
    def create_header_section(self, parent):
        """1️⃣ HEADER SECTION - Space Cowboy Command Center with Cosmic theme"""
        header_frame = tk.Frame(parent, bg=self.colors['cosmic_blue'], height=120)
        header_frame.pack(fill='x', padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        # Left: Spaceship and cosmic elements
        left_decor = tk.Frame(header_frame, bg=self.colors['cosmic_blue'])
        left_decor.pack(side='left', padx=20, pady=20)
        
        spaceship_label = tk.Label(left_decor, text="🚀", font=('Arial', 24), 
                                  bg=self.colors['cosmic_blue'], fg=self.colors['neon_cyan'])
        spaceship_label.pack()
        
        cosmic_label = tk.Label(left_decor, text="⭐ 🌟 ⭐", font=('Arial', 12), 
                               bg=self.colors['cosmic_blue'], fg=self.colors['star_gold'])
        cosmic_label.pack()
        
        # Center: Title and subtitle
        center_frame = tk.Frame(header_frame, bg=self.colors['cosmic_blue'])
        center_frame.pack(expand=True, fill='both', pady=20)
        
        # Main title with Space Cowboy styling
        title_label = tk.Label(center_frame, 
                             text="Welcome to jarvAIs Space Cowboy Command Center",
                             font=('Courier New', 28, 'bold'),
                             bg=self.colors['cosmic_blue'], fg=self.colors['star_gold'])
        title_label.pack(pady=(10, 5))
        
        # Subtitle
        subtitle_label = tk.Label(center_frame,
                                text="Control your computer with hand gestures and voice commands across the cosmic frontier",
                                font=('Arial', 14),
                                bg=self.colors['cosmic_blue'], fg=self.colors['neon_cyan'])
        subtitle_label.pack()
        
        # Cosmic energy underline accent
        energy_frame = tk.Frame(center_frame, bg=self.colors['neon_cyan'], height=3)
        energy_frame.pack(fill='x', padx=50, pady=(10, 0))
        
        # Right: Space cowboy mascot
        right_decor = tk.Frame(header_frame, bg=self.colors['cosmic_blue'])
        right_decor.pack(side='right', padx=20, pady=20)
        
        robot_label = tk.Label(right_decor, text="🤖", font=('Arial', 32), 
                              bg=self.colors['cosmic_blue'], fg=self.colors['plasma_purple'])
        robot_label.pack()
        
        cowboy_label = tk.Label(right_decor, text="🤠", font=('Arial', 16), 
                               bg=self.colors['cosmic_blue'], fg=self.colors['star_gold'])
        cowboy_label.pack()
        
    def create_navigation(self):
        """Create Western-themed navigation bar"""
        nav_frame = tk.Frame(self.root, bg='#2d1b1b', height=60)
        nav_frame.pack(fill='x', padx=0, pady=0)
        nav_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(nav_frame, 
                             text="jarvAIs Western Command Center",
                             font=('Arial', 20, 'bold'),
                             bg='#2d1b1b', fg='#ff8c00')
        title_label.pack(side='left', padx=20, pady=15)
        
        # Navigation buttons
        nav_buttons_frame = tk.Frame(nav_frame, bg='#2d1b1b')
        nav_buttons_frame.pack(side='right', padx=20, pady=10)
        
        page_names = ["Main", "Gestures", "Voice", "Tools", "Settings"]
        self.nav_buttons = []
        
        for i, name in enumerate(page_names):
            btn = tk.Button(nav_buttons_frame,
                          text=name,
                          font=('Arial', 10, 'bold'),
                          bg='#8b4513', fg='#ff8c00',
                          command=lambda i=i: self.show_page(i),
                          width=12, height=2,
                          relief='flat', bd=0)
            btn.pack(side='left', padx=5)
            self.nav_buttons.append(btn)
        
        # Status indicators
        status_frame = tk.Frame(nav_frame, bg='#2d1b1b')
        status_frame.pack(side='right', padx=20, pady=10)
        
        self.status_label = tk.Label(status_frame, 
                                   text="Initializing...",
                                   font=('Arial', 10),
                                   bg='#2d1b1b', fg='#ff8c00')
        self.status_label.pack(side='top')
        
        self.fps_label = tk.Label(status_frame, 
                                text="FPS: 0",
                                font=('Arial', 10),
                                bg='#2d1b1b', fg='#90ee90')
        self.fps_label.pack(side='bottom')
        
    def create_main_page(self):
        """2️⃣ MAIN CONTENT AREA - Reorganized layout with camera, gestures, and voice"""
        self.main_page = tk.Frame(self.content_frame, bg=self.colors['deep_space'])
        
        # TOP SECTION: Hand Gesture Instructions with Auto-Scrolling
        self.create_scrolling_gestures_section()
        
        # MIDDLE SECTION: Camera and Voice Controls
        middle_frame = tk.Frame(self.main_page, bg=self.colors['deep_space'])
        middle_frame.pack(fill='both', expand=True, pady=10)
        
        # LEFT PANEL: Live Camera Feed (moved more to the left)
        camera_panel = tk.Frame(middle_frame, bg=self.colors['card_bg'], relief='raised', bd=2, width=500)
        camera_panel.pack(side='left', fill='both', expand=False, padx=(0, 10))
        
        # Camera panel header
        camera_header = tk.Label(camera_panel, text="🚀 Live Camera Feed", 
                               font=('Courier New', 18, 'bold'),
                               bg=self.colors['card_bg'], fg=self.colors['star_gold'])
        camera_header.pack(pady=(20, 10))
        
        # Camera canvas with Space Cowboy styling (smaller size)
        self.camera_canvas = Canvas(camera_panel, width=480, height=360, 
                                  bg=self.colors['dark_bg'], relief='raised', bd=3,
                                  highlightbackground=self.colors['neon_cyan'])
        self.camera_canvas.pack(pady=(0, 20))
        
        # Camera controls
        camera_controls = tk.Frame(camera_panel, bg=self.colors['card_bg'])
        camera_controls.pack(fill='x', pady=(0, 15))
        
        self.start_button = tk.Button(camera_controls,
                                    text="🚀 ACTIVATE JarvAIs",
                                    font=('Arial', 14, 'bold'),
                                    bg=self.colors['neon_cyan'], fg=self.colors['deep_space'],
                                    command=self.start_gesture_control,
                                    width=20, height=2,
                                    relief='raised', bd=2)
        self.start_button.pack(side='left', padx=(20, 10))
        
        self.stop_button = tk.Button(camera_controls,
                                   text="⛔ DEACTIVATE",
                                   font=('Arial', 14, 'bold'),
                                   bg='#696969', fg='#ffffff',
                                   command=self.stop_gesture_control,
                                   state='disabled', width=20, height=2,
                                   relief='raised', bd=2)
        self.stop_button.pack(side='left', padx=(0, 20))
        
        # Camera status
        camera_status = tk.Label(camera_panel, 
                               text="Camera ready — FPS: 0.00",
                               font=('Arial', 12),
                               bg=self.colors['card_bg'], fg=self.colors['star_gold'])
        camera_status.pack(pady=(0, 20))
        
        # Gesture display
        self.gesture_display = tk.Label(camera_panel, 
                                      text="Gesture: NONE",
                                      font=('Arial', 14, 'bold'),
                                      bg=self.colors['card_bg'], fg=self.colors['neon_cyan'])
        self.gesture_display.pack(pady=(0, 20))
        
        # RIGHT PANEL: Voice Controls and Chatbot
        voice_panel = tk.Frame(middle_frame, bg=self.colors['card_bg'], relief='raised', bd=2, width=400)
        voice_panel.pack(side='right', fill='both', expand=True)
        
        # Voice Control Section
        voice_header = tk.Label(voice_panel, text="🎤 Voice Control", 
                               font=('Courier New', 18, 'bold'),
                               bg=self.colors['card_bg'], fg=self.colors['star_gold'])
        voice_header.pack(pady=(20, 10))
        
        # Voice control button
        self.voice_button = tk.Button(voice_panel,
                                    text="🎤 VOICE ON",
                                    font=('Arial', 12, 'bold'),
                                    bg=self.colors['neon_cyan'], fg=self.colors['deep_space'],
                                    command=self.toggle_jarvis,
                                    width=20, height=2,
                                    relief='raised', bd=2)
        self.voice_button.pack(pady=10)
        
        # Voice transcription display
        self.voice_display = tk.Text(voice_panel, 
                                   height=6, width=45,
                                   font=('Arial', 10),
                                   bg=self.colors['dark_bg'], fg=self.colors['star_gold'],
                                   wrap=tk.WORD, relief='sunken', bd=2)
        self.voice_display.pack(pady=10, padx=10)
        
        # Chatbot Section
        chatbot_header = tk.Label(voice_panel, text="💬 Chatbot", 
                                 font=('Courier New', 16, 'bold'),
                                 bg=self.colors['card_bg'], fg=self.colors['star_gold'])
        chatbot_header.pack(pady=(20, 10))
        
        # Chat messages area
        self.chat_messages = tk.Text(voice_panel, 
                                    height=8, width=45,
                                    font=('Arial', 9),
                                    bg=self.colors['dark_bg'], fg=self.colors['neon_cyan'],
                                    wrap=tk.WORD, relief='sunken', bd=2)
        self.chat_messages.pack(pady=10, padx=10)
        
        # Chat input area
        chat_input_frame = tk.Frame(voice_panel, bg=self.colors['card_bg'])
        chat_input_frame.pack(fill='x', padx=10, pady=(0, 20))
        
        self.chat_input = tk.Entry(chat_input_frame, 
                                  font=('Arial', 10),
                                  bg=self.colors['dark_bg'], fg=self.colors['star_gold'],
                                  relief='sunken', bd=2)
        self.chat_input.pack(side='left', fill='x', expand=True, padx=(0, 5))
        self.chat_input.bind('<Return>', self.send_chat_message)
        
        self.send_chat_btn = tk.Button(chat_input_frame,
                                     text="Send",
                                     font=('Arial', 10, 'bold'),
                                     bg=self.colors['neon_cyan'], fg=self.colors['deep_space'],
                                     command=self.send_chat_message,
                                     width=8, height=1,
                                     relief='raised', bd=2)
        self.send_chat_btn.pack(side='right')
        
    def create_scrolling_gestures_section(self):
        """Create auto-scrolling gesture images section above the camera"""
        gestures_frame = tk.Frame(self.main_page, bg=self.colors['card_bg'], relief='raised', bd=2, height=120)
        gestures_frame.pack(fill='x', padx=10, pady=(0, 10))
        gestures_frame.pack_propagate(False)
        
        # Header
        header = tk.Label(gestures_frame, text="🚀 Hand Gesture Reference", 
                         font=('Courier New', 16, 'bold'),
                         bg=self.colors['card_bg'], fg=self.colors['star_gold'])
        header.pack(pady=(10, 5))
        
        # Scrolling canvas for gesture images
        self.gesture_scroll_canvas = Canvas(gestures_frame, height=80, 
                                           bg=self.colors['dark_bg'], 
                                           highlightthickness=0)
        self.gesture_scroll_canvas.pack(fill='x', padx=10, pady=(0, 10))
        
        # Initialize scrolling animation
        self.scroll_position = 0
        self.scroll_direction = 1
        self.animate_gesture_scroll()
        
    def animate_gesture_scroll(self):
        """Animate the scrolling gesture images"""
        self.gesture_scroll_canvas.delete("all")
        
        # Calculate positions for gesture images
        canvas_width = self.gesture_scroll_canvas.winfo_width()
        if canvas_width <= 1:  # Canvas not yet sized
            self.root.after(100, self.animate_gesture_scroll)
            return
            
        # Create gesture images with labels
        gesture_data = [
            ("👆 Move", self.colors['neon_cyan']),
            ("👆👆 Click", self.colors['star_gold']),
            ("👍 Right Click", self.colors['neon_cyan']),
            ("✊ Scroll Up", self.colors['star_gold']),
            ("✋ Scroll Down", self.colors['neon_cyan']),
            ("🖕 Speech", self.colors['star_gold']),
            ("✋ Drag", self.colors['neon_cyan']),
            ("📺 YouTube", self.colors['star_gold'])
        ]
        
        # Draw scrolling gesture items
        item_width = 120
        spacing = 20
        total_width = len(gesture_data) * (item_width + spacing)
        
        for i, (gesture_text, color) in enumerate(gesture_data):
            x = (i * (item_width + spacing) - self.scroll_position) % (total_width + canvas_width)
            
            # Draw gesture item background
            self.gesture_scroll_canvas.create_rectangle(
                x, 10, x + item_width, 70,
                fill=color, outline=self.colors['deep_space'], width=2
            )
            
            # Draw gesture text
            self.gesture_scroll_canvas.create_text(
                x + item_width//2, 40,
                text=gesture_text,
                font=('Arial', 10, 'bold'),
                fill=self.colors['deep_space']
            )
        
        # Update scroll position
        self.scroll_position += 2 * self.scroll_direction
        
        # Reverse direction when reaching edges
        if self.scroll_position >= total_width:
            self.scroll_direction = -1
        elif self.scroll_position <= 0:
            self.scroll_direction = 1
            
        # Schedule next animation
        self.root.after(50, self.animate_gesture_scroll)
        
    def create_gesture_cards(self, parent_frame):
        """Create gesture instruction cards with Space Cowboy styling and actual images"""
        # Move Pointer Card
        move_card = tk.Frame(parent_frame, bg=self.colors['neon_cyan'], relief='raised', bd=2)
        move_card.pack(fill='x', padx=15, pady=8)
        
        move_title = tk.Label(move_card, text="👆 Move Pointer", 
                             font=('Arial', 14, 'bold'),
                             bg=self.colors['neon_cyan'], fg=self.colors['deep_space'])
        move_title.pack(pady=(12, 5))
        
        # Add gesture image if available
        if 'move' in self.gesture_images:
            move_img_label = tk.Label(move_card, image=self.gesture_images['move'], 
                                     bg=self.colors['neon_cyan'])
            move_img_label.pack(pady=5)
        
        move_desc = tk.Label(move_card, text="Move your index finger to control the mouse cursor", 
                            font=('Arial', 11),
                            bg=self.colors['neon_cyan'], fg=self.colors['deep_space'])
        move_desc.pack(pady=(0, 12))
        
        # Single Click Card
        click_card = tk.Frame(parent_frame, bg=self.colors['star_gold'], relief='raised', bd=2)
        click_card.pack(fill='x', padx=15, pady=8)
        
        click_title = tk.Label(click_card, text="👆 Single Click", 
                              font=('Arial', 14, 'bold'),
                              bg=self.colors['star_gold'], fg=self.colors['deep_space'])
        click_title.pack(pady=(12, 5))
        
        # Add gesture image if available
        if 'click' in self.gesture_images:
            click_img_label = tk.Label(click_card, image=self.gesture_images['click'], 
                                      bg=self.colors['star_gold'])
            click_img_label.pack(pady=5)
        
        click_desc = tk.Label(click_card, text="Point and tap your index finger to left click", 
                             font=('Arial', 11),
                             bg=self.colors['star_gold'], fg=self.colors['deep_space'])
        click_desc.pack(pady=(0, 12))
        
        # Double Click Card
        dclick_card = tk.Frame(parent_frame, bg=self.colors['neon_cyan'], relief='raised', bd=2)
        dclick_card.pack(fill='x', padx=15, pady=8)
        
        dclick_title = tk.Label(dclick_card, text="👆👆 Double Click", 
                               font=('Arial', 14, 'bold'),
                               bg=self.colors['neon_cyan'], fg=self.colors['deep_space'])
        dclick_title.pack(pady=(12, 5))
        
        # Add gesture image if available
        if 'double_click' in self.gesture_images:
            dclick_img_label = tk.Label(dclick_card, image=self.gesture_images['double_click'], 
                                       bg=self.colors['neon_cyan'])
            dclick_img_label.pack(pady=5)
        
        dclick_desc = tk.Label(dclick_card, text="Point and tap twice quickly to double click", 
                              font=('Arial', 11),
                              bg=self.colors['neon_cyan'], fg=self.colors['deep_space'])
        dclick_desc.pack(pady=(0, 12))
        
        # Right Click Card
        rclick_card = tk.Frame(parent_frame, bg=self.colors['star_gold'], relief='raised', bd=2)
        rclick_card.pack(fill='x', padx=15, pady=8)
        
        rclick_title = tk.Label(rclick_card, text="👍 Right Click", 
                               font=('Arial', 14, 'bold'),
                               bg=self.colors['star_gold'], fg=self.colors['deep_space'])
        rclick_title.pack(pady=(12, 5))
        
        # Add gesture image if available
        if 'right_click' in self.gesture_images:
            rclick_img_label = tk.Label(rclick_card, image=self.gesture_images['right_click'], 
                                       bg=self.colors['star_gold'])
            rclick_img_label.pack(pady=5)
        
        rclick_desc = tk.Label(rclick_card, text="Hold your index finger and tap your thumb to right click", 
                              font=('Arial', 11),
                              bg=self.colors['star_gold'], fg=self.colors['deep_space'])
        rclick_desc.pack(pady=(0, 12))
        
        # YouTube Card
        youtube_card = tk.Frame(parent_frame, bg=self.colors['neon_cyan'], relief='raised', bd=2)
        youtube_card.pack(fill='x', padx=15, pady=8)
        
        youtube_title = tk.Label(youtube_card, text="📺 YouTube", 
                               font=('Arial', 14, 'bold'),
                               bg=self.colors['neon_cyan'], fg=self.colors['deep_space'])
        youtube_title.pack(pady=(12, 5))
        
        # Add gesture image if available
        if 'youtube' in self.gesture_images:
            youtube_img_label = tk.Label(youtube_card, image=self.gesture_images['youtube'], 
                                       bg=self.colors['neon_cyan'])
            youtube_img_label.pack(pady=5)
        
        youtube_desc = tk.Label(youtube_card, text="Touch pointer finger and thumb together to open YouTube in Chrome", 
                              font=('Arial', 11),
                              bg=self.colors['neon_cyan'], fg=self.colors['deep_space'])
        youtube_desc.pack(pady=(0, 12))
        
    def create_footer_navigation(self, parent):
        """3️⃣ FOOTER / NAVIGATION BAR - Space Cowboy theme"""
        footer_frame = tk.Frame(parent, bg=self.colors['card_bg'], height=60)
        footer_frame.pack(fill='x', side='bottom', padx=0, pady=0)
        footer_frame.pack_propagate(False)
        
        # Left: App name and theme
        footer_left = tk.Label(footer_frame, 
                              text="jarvAIs Space Cowboy Command Center — Cosmic Frontier",
                              font=('Arial', 12, 'bold'),
                              bg=self.colors['card_bg'], fg=self.colors['neon_cyan'])
        footer_left.pack(side='left', padx=20, pady=15)
        
        # Center: Status indicators
        status_frame = tk.Frame(footer_frame, bg=self.colors['card_bg'])
        status_frame.pack(side='left', expand=True, fill='x', padx=20)
        
        self.camera_status_text = tk.Label(status_frame, 
                                         text="Camera ready",
                                         font=('Arial', 11),
                                         bg=self.colors['card_bg'], fg=self.colors['star_gold'])
        self.camera_status_text.pack(side='left', padx=10)
        
        self.fps_counter = tk.Label(status_frame, 
                                  text="FPS: 00.00",
                                  font=('Arial', 11),
                                  bg=self.colors['card_bg'], fg=self.colors['star_gold'])
        self.fps_counter.pack(side='left', padx=10)
        
        # Right: Navigation tabs
        nav_tabs_frame = tk.Frame(footer_frame, bg=self.colors['card_bg'])
        nav_tabs_frame.pack(side='right', padx=20, pady=10)
        
        tab_names = ["Main", "Settings"]
        self.nav_tabs = []
        
        for i, name in enumerate(tab_names):
            tab_btn = tk.Button(nav_tabs_frame,
                              text=name,
                              font=('Arial', 10, 'bold'),
                              bg=self.colors['neon_cyan'], fg=self.colors['deep_space'],
                              command=lambda i=i: self.show_page(i),
                              width=10, height=1,
                              relief='raised', bd=1)
            tab_btn.pack(side='left', padx=3)
            self.nav_tabs.append(tab_btn)
        
    def create_gestures_page(self):
        """Create the gestures reference page with Space Cowboy theme"""
        self.gestures_page = tk.Frame(self.content_frame, bg=self.colors['deep_space'])
        
        # Page title
        title = tk.Label(self.gestures_page,
                        text="🚀 Hand Gesture Reference",
                        font=('Courier New', 28, 'bold'),
                        bg=self.colors['deep_space'], fg=self.colors['star_gold'])
        title.pack(pady=(0, 30))
        
        # Gesture grid
        gestures_frame = tk.Frame(self.gestures_page, bg=self.colors['deep_space'])
        gestures_frame.pack(fill='both', expand=True)
        
        gestures = [
            ("👆 Index Finger", "Move Mouse", self.colors['neon_cyan']),
            ("👆👆 Index + Middle", "Left Click", self.colors['star_gold']),
            ("👍 Thumb", "Right Click", self.colors['neon_cyan']),
            ("✊ Fist", "Scroll Up", self.colors['star_gold']),
            ("✋ Open Palm", "Scroll Down", self.colors['neon_cyan']),
            ("🖕 Middle Finger", "Speech to Text", self.colors['star_gold'])
        ]
        
        for i, (gesture, action, color) in enumerate(gestures):
            row = i // 2
            col = i % 2
            
            gesture_frame = tk.Frame(gestures_frame, bg=self.colors['card_bg'], relief='raised', bd=2)
            gesture_frame.grid(row=row, column=col, padx=20, pady=20, sticky='nsew')
            
            tk.Label(gesture_frame, text=gesture, font=('Arial', 16, 'bold'),
                    bg=self.colors['card_bg'], fg=color).pack(pady=(20, 10))
            
            tk.Label(gesture_frame, text=action, font=('Arial', 12),
                    bg=self.colors['card_bg'], fg=self.colors['star_gold']).pack(pady=(0, 20))
        
        gestures_frame.grid_columnconfigure(0, weight=1)
        gestures_frame.grid_columnconfigure(1, weight=1)
        
    def create_voice_page(self):
        """4️⃣ VOICE PAGE - Chatbot Integration with Desert Night theme"""
        self.voice_page = tk.Frame(self.content_frame, bg=self.colors['deep_space'])
        
        # Page title
        title = tk.Label(self.voice_page,
                        text="🎤 Voice Command Center",
                        font=('Courier New', 28, 'bold'),
                        bg=self.colors['deep_space'], fg=self.colors['star_gold'])
        title.pack(pady=(0, 20))
        
        # Voice controls section
        voice_control_frame = tk.Frame(self.voice_page, bg=self.colors['card_bg'], relief='raised', bd=2)
        voice_control_frame.pack(fill='x', pady=(0, 15))
        
        tk.Label(voice_control_frame, text="🤖 JARVIS Voice Control", 
                font=('Courier New', 20, 'bold'), 
                bg=self.colors['card_bg'], fg=self.colors['star_gold']).pack(pady=20)
        
        # Voice control buttons
        voice_buttons_frame = tk.Frame(voice_control_frame, bg=self.colors['card_bg'])
        voice_buttons_frame.pack(pady=(0, 15))
        
        self.voice_button = tk.Button(voice_buttons_frame,
                                    text="🎤 VOICE ON",
                                    font=('Arial', 14, 'bold'),
                                    bg=self.colors['neon_cyan'], fg=self.colors['deep_space'],
                                    command=self.toggle_jarvis,
                                    width=15, height=2,
                                    relief='raised', bd=2)
        self.voice_button.pack(side='left', padx=10)
        
        # Voice transcription display
        self.voice_display = tk.Text(voice_control_frame, 
                                   height=8, width=90,
                                   font=('Arial', 11),
                                   bg=self.colors['dark_bg'], fg=self.colors['star_gold'],
                                   wrap=tk.WORD, relief='sunken', bd=2)
        self.voice_display.pack(pady=(0, 20), padx=20)
        
        # Chatbot Integration Section
        chatbot_frame = tk.Frame(self.voice_page, bg=self.colors['card_bg'], relief='raised', bd=2)
        chatbot_frame.pack(fill='both', expand=True)
        
        tk.Label(chatbot_frame, text="💬 Chatbot Integration", 
                font=('Courier New', 18, 'bold'), 
                bg=self.colors['card_bg'], fg=self.colors['star_gold']).pack(pady=20)
        
        # Chat messages area
        chat_messages_frame = tk.Frame(chatbot_frame, bg=self.colors['card_bg'])
        chat_messages_frame.pack(fill='both', expand=True, padx=20, pady=(0, 15))
        
        self.chat_messages = tk.Text(chatbot_frame, 
                                    height=12, width=90,
                                    font=('Arial', 10),
                                    bg=self.colors['dark_bg'], fg=self.colors['neon_cyan'],
                                    wrap=tk.WORD, relief='sunken', bd=2)
        self.chat_messages.pack(pady=(0, 15), padx=20)
        
        # Chat input area
        chat_input_frame = tk.Frame(chatbot_frame, bg=self.colors['card_bg'])
        chat_input_frame.pack(fill='x', padx=20, pady=(0, 20))
        
        self.chat_input = tk.Entry(chat_input_frame, 
                                  font=('Arial', 12),
                                  bg=self.colors['dark_bg'], fg=self.colors['star_gold'],
                                  relief='sunken', bd=2)
        self.chat_input.pack(side='left', fill='x', expand=True, padx=(0, 10))
        self.chat_input.bind('<Return>', self.send_chat_message)
        
        self.send_chat_btn = tk.Button(chat_input_frame,
                                     text="Send",
                                     font=('Arial', 12, 'bold'),
                                     bg=self.colors['neon_cyan'], fg=self.colors['deep_space'],
                                     command=self.send_chat_message,
                                     width=10, height=1,
                                     relief='raised', bd=2)
        self.send_chat_btn.pack(side='right')
        
        # Commands reference
        commands_frame = tk.Frame(self.voice_page, bg=self.colors['card_bg'], relief='raised', bd=2)
        commands_frame.pack(fill='x', pady=(15, 0))
        
        tk.Label(commands_frame, text="📋 Available Voice Commands", 
                font=('Courier New', 16, 'bold'), 
                bg=self.colors['card_bg'], fg=self.colors['star_gold']).pack(pady=15)
        
        commands_text = """
        🎯 Basic Commands: "JARVIS calculator", "JARVIS notepad", "JARVIS browser"
        ⏰ Time Commands: "JARVIS time", "JARVIS date", "JARVIS joke"
        🔊 Audio Commands: "JARVIS volume up", "JARVIS volume down", "JARVIS mute"
        🪟 Window Commands: "JARVIS minimize", "JARVIS close", "JARVIS help"
        """
        
        tk.Label(commands_frame, text=commands_text, font=('Arial', 11),
                bg=self.colors['card_bg'], fg=self.colors['neon_cyan'], 
                justify='left').pack(pady=(0, 15))
        
    def create_tools_page(self):
        """Create the tools page with Space Cowboy theme"""
        self.tools_page = tk.Frame(self.content_frame, bg=self.colors['deep_space'])
        
        # Page title
        title = tk.Label(self.tools_page,
                        text="🛠️ Space Tools & Applications",
                        font=('Courier New', 28, 'bold'),
                        bg=self.colors['deep_space'], fg=self.colors['star_gold'])
        title.pack(pady=(0, 30))
        
        # Tools grid
        tools_frame = tk.Frame(self.tools_page, bg=self.colors['deep_space'])
        tools_frame.pack(fill='both', expand=True)
        
        tools = [
            ("🧮 Calculator", self._open_calculator, self.colors['neon_cyan']),
            ("📝 Notepad", self._open_notepad, self.colors['star_gold']),
            ("🌐 Browser", self._open_browser, self.colors['neon_cyan']),
            ("🎨 Paint", self._open_paint, self.colors['star_gold']),
            ("📊 Excel", self._open_excel, self.colors['neon_cyan']),
            ("📄 Word", self._open_word, self.colors['star_gold'])
        ]
        
        for i, (name, command, color) in enumerate(tools):
            row = i // 3
            col = i % 3
            
            tool_frame = tk.Frame(tools_frame, bg=self.colors['card_bg'], relief='raised', bd=2)
            tool_frame.grid(row=row, column=col, padx=20, pady=20, sticky='nsew')
            
            tk.Button(tool_frame, text=name, command=command,
                     font=('Arial', 14, 'bold'), bg=color, fg=self.colors['deep_space'],
                     width=15, height=3, relief='raised', bd=2).pack(expand=True, fill='both', padx=20, pady=20)
        
        tools_frame.grid_columnconfigure(0, weight=1)
        tools_frame.grid_columnconfigure(1, weight=1)
        tools_frame.grid_columnconfigure(2, weight=1)
        
    def create_settings_page(self):
        """Create the settings page with Space Cowboy theme"""
        self.settings_page = tk.Frame(self.content_frame, bg=self.colors['deep_space'])
        
        # Page title
        title = tk.Label(self.settings_page,
                        text="⚙️ System Settings",
                        font=('Courier New', 28, 'bold'),
                        bg=self.colors['deep_space'], fg=self.colors['star_gold'])
        title.pack(pady=(0, 30))
        
        # Settings content
        settings_frame = tk.Frame(self.settings_page, bg=self.colors['card_bg'], relief='raised', bd=2)
        settings_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        tk.Label(settings_frame, text="🤖 jarvAIs Configuration", 
                font=('Courier New', 20, 'bold'), 
                bg=self.colors['card_bg'], fg=self.colors['star_gold']).pack(pady=20)
        
        # Sensitivity Controls Section
        sensitivity_frame = tk.Frame(settings_frame, bg=self.colors['card_bg'])
        sensitivity_frame.pack(fill='x', padx=20, pady=20)
        
        # Mouse Sensitivity Control
        mouse_frame = tk.Frame(sensitivity_frame, bg=self.colors['neon_cyan'], relief='raised', bd=2)
        mouse_frame.pack(fill='x', pady=10)
        
        tk.Label(mouse_frame, text="🖱️ Mouse Sensitivity", 
                font=('Arial', 16, 'bold'),
                bg=self.colors['neon_cyan'], fg=self.colors['deep_space']).pack(pady=(15, 10))
        
        mouse_control_frame = tk.Frame(mouse_frame, bg=self.colors['neon_cyan'])
        mouse_control_frame.pack(fill='x', padx=20, pady=(0, 15))
        
        tk.Label(mouse_control_frame, text="Smoothing Factor:", 
                font=('Arial', 12),
                bg=self.colors['neon_cyan'], fg=self.colors['deep_space']).pack(side='left')
        
        self.mouse_sensitivity_var = tk.IntVar(value=self.config.smoothing_factor)
        self.mouse_sensitivity_scale = tk.Scale(mouse_control_frame, 
                                               from_=1, to=20, 
                                               orient='horizontal',
                                               variable=self.mouse_sensitivity_var,
                                               bg=self.colors['neon_cyan'], 
                                               fg=self.colors['deep_space'],
                                               highlightthickness=0,
                                               command=self.update_mouse_sensitivity)
        self.mouse_sensitivity_scale.pack(side='right', fill='x', expand=True, padx=(10, 0))
        
        # Scroll Sensitivity Control
        scroll_frame = tk.Frame(sensitivity_frame, bg=self.colors['star_gold'], relief='raised', bd=2)
        scroll_frame.pack(fill='x', pady=10)
        
        tk.Label(scroll_frame, text="📜 Scroll Sensitivity", 
                font=('Arial', 16, 'bold'),
                bg=self.colors['star_gold'], fg=self.colors['deep_space']).pack(pady=(15, 10))
        
        scroll_control_frame = tk.Frame(scroll_frame, bg=self.colors['star_gold'])
        scroll_control_frame.pack(fill='x', padx=20, pady=(0, 15))
        
        tk.Label(scroll_control_frame, text="Scroll Speed:", 
                font=('Arial', 12),
                bg=self.colors['star_gold'], fg=self.colors['deep_space']).pack(side='left')
        
        self.scroll_sensitivity_var = tk.IntVar(value=abs(self.config.scroll_up_speed))
        self.scroll_sensitivity_scale = tk.Scale(scroll_control_frame, 
                                                from_=10, to=120, 
                                                orient='horizontal',
                                                variable=self.scroll_sensitivity_var,
                                                bg=self.colors['star_gold'], 
                                                fg=self.colors['deep_space'],
                                                highlightthickness=0,
                                                command=self.update_scroll_sensitivity)
        self.scroll_sensitivity_scale.pack(side='right', fill='x', expand=True, padx=(10, 0))
        
        # Camera Settings Section
        camera_frame = tk.Frame(sensitivity_frame, bg=self.colors['neon_cyan'], relief='raised', bd=2)
        camera_frame.pack(fill='x', pady=10)
        
        tk.Label(camera_frame, text="📹 Camera Settings", 
                font=('Arial', 16, 'bold'),
                bg=self.colors['neon_cyan'], fg=self.colors['deep_space']).pack(pady=(15, 10))
        
        camera_control_frame = tk.Frame(camera_frame, bg=self.colors['neon_cyan'])
        camera_control_frame.pack(fill='x', padx=20, pady=(0, 15))
        
        # FPS Control
        fps_frame = tk.Frame(camera_control_frame, bg=self.colors['neon_cyan'])
        fps_frame.pack(fill='x', pady=5)
        
        tk.Label(fps_frame, text="Camera FPS:", 
                font=('Arial', 12),
                bg=self.colors['neon_cyan'], fg=self.colors['deep_space']).pack(side='left')
        
        self.fps_var = tk.IntVar(value=self.config.camera_fps)
        self.fps_scale = tk.Scale(fps_frame, 
                                 from_=15, to=60, 
                                 orient='horizontal',
                                 variable=self.fps_var,
                                 bg=self.colors['neon_cyan'], 
                                 fg=self.colors['deep_space'],
                                 highlightthickness=0,
                                 command=self.update_camera_fps)
        self.fps_scale.pack(side='right', fill='x', expand=True, padx=(10, 0))
        
        # Stability Control
        stability_frame = tk.Frame(camera_control_frame, bg=self.colors['neon_cyan'])
        stability_frame.pack(fill='x', pady=5)
        
        tk.Label(stability_frame, text="Click Stability:", 
                font=('Arial', 12),
                bg=self.colors['neon_cyan'], fg=self.colors['deep_space']).pack(side='left')
        
        self.stability_var = tk.IntVar(value=self.config.stability_threshold)
        self.stability_scale = tk.Scale(stability_frame, 
                                       from_=5, to=20, 
                                       orient='horizontal',
                                       variable=self.stability_var,
                                       bg=self.colors['neon_cyan'], 
                                       fg=self.colors['deep_space'],
                                       highlightthickness=0,
                                       command=self.update_stability)
        self.stability_scale.pack(side='right', fill='x', expand=True, padx=(10, 0))
        
        # Reset Button
        reset_frame = tk.Frame(settings_frame, bg=self.colors['card_bg'])
        reset_frame.pack(fill='x', padx=20, pady=20)
        
        reset_button = tk.Button(reset_frame,
                               text="🔄 Reset to Defaults",
                               font=('Arial', 14, 'bold'),
                               bg=self.colors['accent'], fg=self.colors['deep_space'],
                               command=self.reset_settings,
                               width=20, height=2,
                               relief='raised', bd=2)
        reset_button.pack()
        
    def update_mouse_sensitivity(self, value):
        """Update mouse sensitivity (smoothing factor)"""
        self.config.smoothing_factor = int(value)
        if hasattr(self, 'recognizer') and self.recognizer:
            self.recognizer.config.smoothing_factor = int(value)
        print(f"Mouse sensitivity updated to: {value}")
        
    def update_scroll_sensitivity(self, value):
        """Update scroll sensitivity"""
        scroll_speed = int(value)
        self.config.scroll_up_speed = scroll_speed
        self.config.scroll_down_speed = -scroll_speed
        if hasattr(self, 'recognizer') and self.recognizer:
            self.recognizer.config.scroll_up_speed = scroll_speed
            self.recognizer.config.scroll_down_speed = -scroll_speed
        print(f"Scroll sensitivity updated to: {scroll_speed}")
        
    def update_camera_fps(self, value):
        """Update camera FPS"""
        self.config.camera_fps = int(value)
        if self.cap:
            self.cap.set(cv2.CAP_PROP_FPS, int(value))
        print(f"Camera FPS updated to: {value}")
        
    def update_stability(self, value):
        """Update click stability threshold"""
        self.config.stability_threshold = int(value)
        if hasattr(self, 'recognizer') and self.recognizer:
            self.recognizer.config.stability_threshold = int(value)
        print(f"Click stability updated to: {value}")
        
    def reset_settings(self):
        """Reset all settings to default values"""
        # Reset to default configuration
        self.config = Config()
        
        # Update UI controls
        if hasattr(self, 'mouse_sensitivity_scale'):
            self.mouse_sensitivity_var.set(self.config.smoothing_factor)
        if hasattr(self, 'scroll_sensitivity_scale'):
            self.scroll_sensitivity_var.set(abs(self.config.scroll_up_speed))
        if hasattr(self, 'fps_scale'):
            self.fps_var.set(self.config.camera_fps)
        if hasattr(self, 'stability_scale'):
            self.stability_var.set(self.config.stability_threshold)
            
        # Update recognizer if it exists
        if hasattr(self, 'recognizer') and self.recognizer:
            self.recognizer.config = self.config
            
        print("Settings reset to defaults")
        
    def show_page(self, page_index):
        """Show the specified page with Space Cowboy navigation"""
        # Hide all pages
        for page in [self.main_page, self.settings_page]:
            page.pack_forget()
        
        # Update navigation tabs
        for i, btn in enumerate(self.nav_tabs):
            if i == page_index:
                btn.config(bg=self.colors['star_gold'], fg=self.colors['deep_space'])
            else:
                btn.config(bg=self.colors['neon_cyan'], fg=self.colors['deep_space'])
        
        # Show selected page
        pages = [self.main_page, self.settings_page]
        pages[page_index].pack(fill='both', expand=True)
        self.current_page = page_index
        
    def send_chat_message(self, event=None):
        """Send message to chatbot integration"""
        message = self.chat_input.get().strip()
        if not message:
            return
            
        # Add user message to chat
        self.chat_messages.insert(tk.END, f"👤 You: {message}\n")
        self.chat_messages.see(tk.END)
        
        # Clear input
        self.chat_input.delete(0, tk.END)
        
        # Simulate chatbot response (replace with actual chatbot integration)
        self.root.after(1000, lambda: self.receive_bot_response(message))
        
    def receive_bot_response(self, user_message):
        """Receive response from chatbot (placeholder for integration)"""
        # This is where you would integrate with your chatbot API
        responses = [
            f"🤖 JARVIS: I understand you said '{user_message}'. How can I help you?",
            f"🤖 JARVIS: Command received: '{user_message}'. Processing...",
            f"🤖 JARVIS: '{user_message}' - I'm ready to assist with that task.",
            f"🤖 JARVIS: Understood '{user_message}'. What would you like me to do next?"
        ]
        
        import random
        response = random.choice(responses)
        self.chat_messages.insert(tk.END, f"{response}\n")
        self.chat_messages.see(tk.END)
        
    def start_camera(self):
        """Initialize camera and start processing"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.status_label.config(text="Camera Error", fg='#ff0000')
                return
                
            # Configure camera
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.camera_fps)
            
            # Initialize recognizer
            self.recognizer = HandGestureRecognizer(self.config)
            
            self.camera_status_text.config(text="Camera Ready", fg=self.colors['star_gold'])
            
        except Exception as e:
            self.status_label.config(text=f"Error: {e}", fg='#ff0000')
            print(f"Camera initialization error: {e}")
            
    def start_gesture_control(self):
        """Start the gesture control session"""
        if not self.cap or not self.recognizer:
            print("Camera not ready")
            return
            
        self.running = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.camera_status_text.config(text="jarvAIs Active", fg=self.colors['star_gold'])
        
        # Start camera processing thread
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
        
        # Start display update
        self.update_display()
        
    def stop_gesture_control(self):
        """Stop the gesture control session"""
        self.running = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.camera_status_text.config(text="jarvAIs Stopped", fg=self.colors['neon_cyan'])
        
    def camera_loop(self):
        """Main camera processing loop"""
        while self.running:
            try:
                success, frame = self.cap.read()
                if not success:
                    continue
                    
                # Process frame
                frame = self.recognizer.detect_hands(frame)
                landmarks, _ = self.recognizer.get_hand_landmarks(frame)
                
                # Detect and execute gestures
                if len(landmarks) >= 21:  # Full hand detected
                    fingers = self.recognizer.detect_finger_states(landmarks)
                    gesture = self.recognizer.classify_gesture(fingers)
                    self.recognizer.execute_gesture(gesture, landmarks, frame)
                    
                    # Update gesture display
                    self.root.after(0, lambda g=gesture: self.gesture_display.config(text=g.value.upper()))
                else:
                    gesture = GestureType.NONE
                    self.root.after(0, lambda: self.gesture_display.config(text="NONE"))
                
                # Calculate FPS
                current_time = time.time()
                fps = 1 / (current_time - self.recognizer.previous_time) if self.recognizer.previous_time > 0 else 0
                self.recognizer.previous_time = current_time
                
                # Update FPS display
                self.root.after(0, lambda f=fps: self.fps_counter.config(text=f"FPS: {int(f):.2f}"))
                
                # Add frame to queue
                if not self.frame_queue.full():
                    self.frame_queue.put((frame, gesture))
                    
            except Exception as e:
                print(f"Camera loop error: {e}")
                time.sleep(0.1)
                
    def update_display(self):
        """Update the camera display"""
        if not self.frame_queue.empty():
            frame, gesture = self.frame_queue.get()
            
            # Convert frame for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tk = ImageTk.PhotoImage(frame_pil)
            
            # Update canvas
            self.camera_canvas.delete("all")
            self.camera_canvas.create_image(240, 180, image=frame_tk)
            self.camera_canvas.image = frame_tk  # Keep reference
            
        # Schedule next update
        if self.running:
            self.root.after(33, self.update_display)  # ~30 FPS
            
    def _jarvis_callback(self, message):
        """JARVIS callback for voice updates"""
        self.voice_display.insert(tk.END, f"{message}\n")
        self.voice_display.see(tk.END)
        
    def toggle_jarvis(self):
        """Toggle JARVIS voice control"""
        if not self.jarvis.listening:
            if self.jarvis.start_listening():
                self.voice_button.config(text="🎤 VOICE OFF", bg=self.colors['accent'])
                self.voice_display.insert(tk.END, "🤖 jarvAIs voice control activated!\n")
                self.voice_display.see(tk.END)
            else:
                self.voice_display.insert(tk.END, "❌ jarvAIs microphone not available!\n")
                self.voice_display.see(tk.END)
        else:
            self.jarvis.stop_listening()
            self.voice_button.config(text="🎤 VOICE ON", bg=self.colors['neon_cyan'])
            self.voice_display.insert(tk.END, "⏹️ jarvAIs voice control stopped.\n")
            self.voice_display.see(tk.END)
    
    # Tool methods
    def _open_calculator(self):
        """Open calculator"""
        subprocess.Popen("calc.exe")
        self.voice_display.insert(tk.END, "🧮 Calculator opened\n")
        self.voice_display.see(tk.END)
        
    def _open_notepad(self):
        """Open notepad"""
        subprocess.Popen("notepad.exe")
        self.voice_display.insert(tk.END, "📝 Notepad opened\n")
        self.voice_display.see(tk.END)
        
    def _open_browser(self):
        """Open web browser"""
        webbrowser.open("https://www.google.com")
        self.voice_display.insert(tk.END, "🌐 Browser opened\n")
        self.voice_display.see(tk.END)
        
    def _open_paint(self):
        """Open Paint"""
        subprocess.Popen("mspaint.exe")
        self.voice_display.insert(tk.END, "🎨 Paint opened\n")
        self.voice_display.see(tk.END)
        
    def _open_excel(self):
        """Open Excel"""
        subprocess.Popen("excel.exe")
        self.voice_display.insert(tk.END, "📊 Excel opened\n")
        self.voice_display.see(tk.END)
        
    def _open_word(self):
        """Open Word"""
        subprocess.Popen("winword.exe")
        self.voice_display.insert(tk.END, "📄 Word opened\n")
        self.voice_display.see(tk.END)
        
    def on_closing(self):
        """Handle application closing"""
        self.running = False
        if self.cap:
            self.cap.release()
        if self.jarvis:
            self.jarvis.stop_listening()
        self.root.destroy()
        
    def run(self):
        """Run the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()


def run_command_line():
    """Run the command-line version"""
    print("jarvAIs - Advanced Hand Gesture Recognition System")
    print("=" * 60)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="jarvAIs - Advanced Hand Gesture Recognition")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    parser.add_argument("--fps", type=int, default=60, help="Camera FPS")
    parser.add_argument("--no-fps", action="store_true", help="Hide FPS display")
    parser.add_argument("--no-gesture-info", action="store_true", help="Hide gesture info")
    parser.add_argument("--sensitivity", type=int, default=8, help="Mouse sensitivity (1-20)")
    
    args = parser.parse_args()
    
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
        print("Error: Could not open camera. Please check camera permissions.")
        print("Tip: Go to System Preferences > Security & Privacy > Camera")
        return 1
        
    # Configure camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)
    cap.set(cv2.CAP_PROP_FPS, config.camera_fps)
    
    # Initialize gesture recognizer
    recognizer = HandGestureRecognizer(config)
    
    print("Starting gesture recognition...")
    print("Gesture Guide:")
    print("   Index finger: Move mouse")
    print("   Index + Middle: Click")
    print("   Thumb: Right click")
    print("   Fist: Scroll up")
    print("   Open palm: Scroll down")
    print("   Middle finger: Speech to text")
    print("   Press 'q' to quit")
    print("-" * 60)
    
    # Main loop
    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Error: Could not read frame from camera")
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
            cv2.imshow("jarvAIs - Hand Gesture Control", frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("jarvAIs session ended. Thank you!")
        
    return 0


def main():
    """Main application entry point"""
    if len(sys.argv) > 1 and sys.argv[1] == '--cli':
        # Command-line mode
        return run_command_line()
    else:
        # GUI mode (default)
        if not GUI_AVAILABLE:
            print("GUI libraries not available. Running in command-line mode.")
            print("To install GUI dependencies: pip install tkinter pillow")
            return run_command_line()
        
        app = jarvAIsGUI()
        app.run()
        return 0


if __name__ == "__main__":
    sys.exit(main())
