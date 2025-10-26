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
    """Complete integrated jarvAIs application with JARVIS voice control"""
    
    def __init__(self):
        if not GUI_AVAILABLE:
            print("GUI libraries not available. Please install tkinter and PIL.")
            sys.exit(1)
            
        self.root = tk.Tk()
        self.root.title("jarvAIs - Space Command Center")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#0a0a1a')
        
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
        
    def create_widgets(self):
        """Create the space-themed GUI interface"""
        # Main container with space background
        main_frame = tk.Frame(self.root, bg='#0a0a1a')
        main_frame.pack(fill='both', expand=True)
        
        # Top navigation bar
        self.create_navigation()
        
        # Main content area
        self.content_frame = tk.Frame(main_frame, bg='#0a0a1a')
        self.content_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Create all pages
        self.create_main_page()
        self.create_gestures_page()
        self.create_voice_page()
        self.create_tools_page()
        self.create_settings_page()
        
        # Show initial page
        self.show_page(0)
        
    def create_starfield(self):
        """Create animated starfield background"""
        self.starfield_canvas = Canvas(self.root, bg='#0a0a1a', highlightthickness=0)
        self.starfield_canvas.place(x=0, y=0, relwidth=1, relheight=1)
        # Send to back
        self.starfield_canvas.lower()
        
        # Create stars
        self.stars = []
        for _ in range(100):
            x = self.root.winfo_screenwidth() * 0.8 * (0.1 + 0.8 * (hash(str(_)) % 1000) / 1000)
            y = self.root.winfo_screenheight() * 0.8 * (0.1 + 0.8 * (hash(str(_ + 1000)) % 1000) / 1000)
            size = 1 + (hash(str(_ + 2000)) % 3)
            self.stars.append({
                'x': x, 'y': y, 'size': size,
                'brightness': 0.3 + 0.7 * (hash(str(_ + 3000)) % 1000) / 1000
            })
        
        self.animate_stars()
        
    def animate_stars(self):
        """Animate the starfield"""
        self.starfield_canvas.delete("all")
        
        for star in self.stars:
            # Twinkling effect
            brightness = star['brightness'] + 0.3 * (hash(str(star['x'] + star['y'])) % 1000) / 1000
            brightness = min(1.0, brightness)
            
            color = f"#{int(100 * brightness):02x}{int(150 * brightness):02x}{int(255 * brightness):02x}"
            
            self.starfield_canvas.create_oval(
                star['x'] - star['size'], star['y'] - star['size'],
                star['x'] + star['size'], star['y'] + star['size'],
                fill=color, outline=color
            )
        
        # Schedule next animation
        self.root.after(100, self.animate_stars)
        
    def create_navigation(self):
        """Create space-themed navigation bar"""
        nav_frame = tk.Frame(self.root, bg='#1a1a2e', height=60)
        nav_frame.pack(fill='x', padx=0, pady=0)
        nav_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(nav_frame, 
                             text="jarvAIs Space Command Center",
                             font=('Arial', 20, 'bold'),
                             bg='#1a1a2e', fg='#4fc3f7')
        title_label.pack(side='left', padx=20, pady=15)
        
        # Navigation buttons
        nav_buttons_frame = tk.Frame(nav_frame, bg='#1a1a2e')
        nav_buttons_frame.pack(side='right', padx=20, pady=10)
        
        page_names = ["Main", "Gestures", "Voice", "Tools", "Settings"]
        self.nav_buttons = []
        
        for i, name in enumerate(page_names):
            btn = tk.Button(nav_buttons_frame,
                          text=name,
                          font=('Arial', 10, 'bold'),
                          bg='#2d2d44', fg='#4fc3f7',
                          command=lambda i=i: self.show_page(i),
                          width=12, height=2,
                          relief='flat', bd=0)
            btn.pack(side='left', padx=5)
            self.nav_buttons.append(btn)
        
        # Status indicators
        status_frame = tk.Frame(nav_frame, bg='#1a1a2e')
        status_frame.pack(side='right', padx=20, pady=10)
        
        self.status_label = tk.Label(status_frame, 
                                   text="Initializing...",
                                   font=('Arial', 10),
                                   bg='#1a1a2e', fg='#4fc3f7')
        self.status_label.pack(side='top')
        
        self.fps_label = tk.Label(status_frame, 
                                text="FPS: 0",
                                font=('Arial', 10),
                                bg='#1a1a2e', fg='#81c784')
        self.fps_label.pack(side='bottom')
        
    def create_main_page(self):
        """Create the main dashboard page"""
        self.main_page = tk.Frame(self.content_frame, bg='#0a0a1a')
        
        # Welcome section
        welcome_frame = tk.Frame(self.main_page, bg='#1a1a2e', relief='raised', bd=2)
        welcome_frame.pack(fill='x', pady=(0, 20))
        
        welcome_label = tk.Label(welcome_frame,
                               text="Welcome to jarvAIs Space Command Center",
                               font=('Arial', 24, 'bold'),
                               bg='#1a1a2e', fg='#4fc3f7')
        welcome_label.pack(pady=20)
        
        # Camera section
        camera_frame = tk.Frame(self.main_page, bg='#1a1a2e', relief='raised', bd=2)
        camera_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        camera_title = tk.Label(camera_frame, text="Live Camera Feed", 
                               font=('Arial', 16, 'bold'),
                               bg='#1a1a2e', fg='#ffffff')
        camera_title.pack(pady=(20, 10))
        
        # Camera canvas with space border
        self.camera_canvas = Canvas(camera_frame, width=640, height=480, 
                                  bg='#000011', relief='raised', bd=3,
                                  highlightbackground='#4fc3f7')
        self.camera_canvas.pack(pady=(0, 20))
        
        # Control buttons
        control_frame = tk.Frame(camera_frame, bg='#1a1a2e')
        control_frame.pack(fill='x', pady=(0, 20))
        
        self.start_button = tk.Button(control_frame,
                                    text="ACTIVATE jarvAIs",
                                    font=('Arial', 12, 'bold'),
                                    bg='#4fc3f7', fg='#000011',
                                    command=self.start_gesture_control,
                                    width=18, height=2)
        self.start_button.pack(side='left', padx=(20, 10))
        
        self.stop_button = tk.Button(control_frame,
                                   text="DEACTIVATE",
                                   font=('Arial', 12, 'bold'),
                                   bg='#f44336', fg='#ffffff',
                                   command=self.stop_gesture_control,
                                   state='disabled', width=18, height=2)
        self.stop_button.pack(side='left', padx=(0, 10))
        
        self.voice_button = tk.Button(control_frame,
                                    text="VOICE ON",
                                    font=('Arial', 12, 'bold'),
                                    bg='#81c784', fg='#000011',
                                    command=self.toggle_jarvis,
                                    width=15, height=2)
        self.voice_button.pack(side='left', padx=(0, 20))
        
        # Gesture display
        gesture_frame = tk.Frame(self.main_page, bg='#1a1a2e', relief='raised', bd=2, width=300)
        gesture_frame.pack(side='right', fill='y')
        gesture_frame.pack_propagate(False)
        
        gesture_title = tk.Label(gesture_frame, text="Current Gesture", 
                                font=('Arial', 16, 'bold'),
                                bg='#1a1a2e', fg='#ffffff')
        gesture_title.pack(pady=(20, 10))
        
        self.gesture_display = tk.Label(gesture_frame, 
                                      text="NONE",
                                      font=('Arial', 24, 'bold'),
                                      bg='#2d2d44', fg='#4fc3f7',
                                      width=12, height=3)
        self.gesture_display.pack(pady=(0, 20))
        
        # Quick stats
        stats_frame = tk.Frame(gesture_frame, bg='#1a1a2e')
        stats_frame.pack(fill='x', padx=20, pady=(0, 20))
        
        tk.Label(stats_frame, text="System Status", font=('Arial', 12, 'bold'),
                bg='#1a1a2e', fg='#ffffff').pack(anchor='w')
        
        self.jarvis_status = tk.Label(stats_frame, 
                                    text="jarvAIs: Ready",
                                    font=('Arial', 10),
                                    bg='#1a1a2e', fg='#81c784')
        self.jarvis_status.pack(anchor='w', pady=(5, 0))
        
    def create_gestures_page(self):
        """Create the gestures reference page"""
        self.gestures_page = tk.Frame(self.content_frame, bg='#0a0a1a')
        
        # Page title
        title = tk.Label(self.gestures_page,
                        text="Hand Gesture Reference",
                        font=('Arial', 24, 'bold'),
                        bg='#0a0a1a', fg='#4fc3f7')
        title.pack(pady=(0, 30))
        
        # Gesture grid
        gestures_frame = tk.Frame(self.gestures_page, bg='#0a0a1a')
        gestures_frame.pack(fill='both', expand=True)
        
        gestures = [
            ("Index Finger", "Move Mouse", "#4fc3f7"),
            ("Index + Middle", "Left Click", "#81c784"),
            ("Thumb", "Right Click", "#ffb74d"),
            ("Fist", "Scroll Up", "#f48fb1"),
            ("Open Palm", "Scroll Down", "#ce93d8"),
            ("Middle Finger", "Speech to Text", "#a5d6a7")
        ]
        
        for i, (gesture, action, color) in enumerate(gestures):
            row = i // 2
            col = i % 2
            
            gesture_frame = tk.Frame(gestures_frame, bg='#1a1a2e', relief='raised', bd=2)
            gesture_frame.grid(row=row, column=col, padx=20, pady=20, sticky='nsew')
            
            tk.Label(gesture_frame, text=gesture, font=('Arial', 16, 'bold'),
                    bg='#1a1a2e', fg=color).pack(pady=(20, 10))
            
            tk.Label(gesture_frame, text=action, font=('Arial', 12),
                    bg='#1a1a2e', fg='#ffffff').pack(pady=(0, 20))
        
        gestures_frame.grid_columnconfigure(0, weight=1)
        gestures_frame.grid_columnconfigure(1, weight=1)
        
    def create_voice_page(self):
        """Create the voice control page"""
        self.voice_page = tk.Frame(self.content_frame, bg='#0a0a1a')
        
        # Page title
        title = tk.Label(self.voice_page,
                        text="Voice Command Center",
                        font=('Arial', 24, 'bold'),
                        bg='#0a0a1a', fg='#4fc3f7')
        title.pack(pady=(0, 30))
        
        # Voice controls
        voice_control_frame = tk.Frame(self.voice_page, bg='#1a1a2e', relief='raised', bd=2)
        voice_control_frame.pack(fill='x', pady=(0, 20))
        
        tk.Label(voice_control_frame, text="JARVIS Voice Control", 
                font=('Arial', 18, 'bold'), bg='#1a1a2e', fg='#ffffff').pack(pady=20)
        
        # Voice transcription
        self.voice_display = tk.Text(voice_control_frame, 
                                   height=10, width=80,
                                   font=('Arial', 11),
                                   bg='#2d2d44', fg='#4fc3f7',
                                   wrap=tk.WORD, relief='sunken', bd=2)
        self.voice_display.pack(pady=(0, 20), padx=20)
        
        # Commands reference
        commands_frame = tk.Frame(self.voice_page, bg='#1a1a2e', relief='raised', bd=2)
        commands_frame.pack(fill='both', expand=True)
        
        tk.Label(commands_frame, text="Available Commands", 
                font=('Arial', 16, 'bold'), bg='#1a1a2e', fg='#ffffff').pack(pady=20)
        
        commands_text = """
        • "JARVIS calculator" - Opens calculator
        • "JARVIS notepad" - Opens notepad  
        • "JARVIS browser" - Opens web browser
        • "JARVIS time" - Tells current time
        • "JARVIS date" - Tells current date
        • "JARVIS joke" - Tells a joke
        • "JARVIS volume up" - Increases volume
        • "JARVIS volume down" - Decreases volume
        • "JARVIS minimize" - Minimizes window
        • "JARVIS close" - Closes window
        """
        
        tk.Label(commands_frame, text=commands_text, font=('Arial', 12),
                bg='#1a1a2e', fg='#81c784', justify='left').pack(pady=(0, 20))
        
    def create_tools_page(self):
        """Create the tools page"""
        self.tools_page = tk.Frame(self.content_frame, bg='#0a0a1a')
        
        # Page title
        title = tk.Label(self.tools_page,
                        text="Space Tools & Applications",
                        font=('Arial', 24, 'bold'),
                        bg='#0a0a1a', fg='#4fc3f7')
        title.pack(pady=(0, 30))
        
        # Tools grid
        tools_frame = tk.Frame(self.tools_page, bg='#0a0a1a')
        tools_frame.pack(fill='both', expand=True)
        
        tools = [
            ("Calculator", self._open_calculator, "#4fc3f7"),
            ("Notepad", self._open_notepad, "#81c784"),
            ("Browser", self._open_browser, "#ffb74d"),
            ("Paint", self._open_paint, "#f48fb1"),
            ("Excel", self._open_excel, "#ce93d8"),
            ("Word", self._open_word, "#a5d6a7")
        ]
        
        for i, (name, command, color) in enumerate(tools):
            row = i // 3
            col = i % 3
            
            tool_frame = tk.Frame(tools_frame, bg='#1a1a2e', relief='raised', bd=2)
            tool_frame.grid(row=row, column=col, padx=20, pady=20, sticky='nsew')
            
            tk.Button(tool_frame, text=name, command=command,
                     font=('Arial', 14, 'bold'), bg=color, fg='#000011',
                     width=15, height=3, relief='flat').pack(expand=True, fill='both', padx=20, pady=20)
        
        tools_frame.grid_columnconfigure(0, weight=1)
        tools_frame.grid_columnconfigure(1, weight=1)
        tools_frame.grid_columnconfigure(2, weight=1)
        
    def create_settings_page(self):
        """Create the settings page"""
        self.settings_page = tk.Frame(self.content_frame, bg='#0a0a1a')
        
        # Page title
        title = tk.Label(self.settings_page,
                        text="System Settings",
                        font=('Arial', 24, 'bold'),
                        bg='#0a0a1a', fg='#4fc3f7')
        title.pack(pady=(0, 30))
        
        # Settings content
        settings_frame = tk.Frame(self.settings_page, bg='#1a1a2e', relief='raised', bd=2)
        settings_frame.pack(fill='both', expand=True)
        
        tk.Label(settings_frame, text="jarvAIs Configuration", 
                font=('Arial', 18, 'bold'), bg='#1a1a2e', fg='#ffffff').pack(pady=20)
        
        # Settings will be added here
        tk.Label(settings_frame, text="Settings panel coming soon...", 
                font=('Arial', 12), bg='#1a1a2e', fg='#81c784').pack(pady=50)
        
    def show_page(self, page_index):
        """Show the specified page"""
        # Hide all pages
        for page in [self.main_page, self.gestures_page, self.voice_page, 
                    self.tools_page, self.settings_page]:
            page.pack_forget()
        
        # Update navigation buttons
        for i, btn in enumerate(self.nav_buttons):
            if i == page_index:
                btn.config(bg='#4fc3f7', fg='#000011')
            else:
                btn.config(bg='#2d2d44', fg='#4fc3f7')
        
        # Show selected page
        pages = [self.main_page, self.gestures_page, self.voice_page, 
                self.tools_page, self.settings_page]
        pages[page_index].pack(fill='both', expand=True)
        self.current_page = page_index
        
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
            
            self.status_label.config(text="Camera Ready", fg='#00ff88')
            
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
        self.status_label.config(text="jarvAIs Active", fg='#00ff88')
        
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
        self.status_label.config(text="jarvAIs Stopped", fg='#ff6b35')
        
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
                self.root.after(0, lambda f=fps: self.fps_label.config(text=f"FPS: {int(f)}"))
                
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
            self.camera_canvas.create_image(320, 240, image=frame_tk)
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
                self.voice_button.config(text="VOICE OFF", bg='#f44336')
                self.jarvis_status.config(text="jarvAIs: Listening...", fg='#00ff88')
                self.voice_display.insert(tk.END, "jarvAIs voice control activated!\n")
                self.voice_display.see(tk.END)
            else:
                self.voice_display.insert(tk.END, "jarvAIs microphone not available!\n")
                self.voice_display.see(tk.END)
        else:
            self.jarvis.stop_listening()
            self.voice_button.config(text="VOICE ON", bg='#81c784')
            self.jarvis_status.config(text="jarvAIs: Stopped", fg='#ff6b35')
            self.voice_display.insert(tk.END, "jarvAIs voice control stopped.\n")
            self.voice_display.see(tk.END)
    
    # Tool methods
    def _open_calculator(self):
        """Open calculator"""
        subprocess.Popen("calc.exe")
        self.voice_display.insert(tk.END, "Calculator opened\n")
        self.voice_display.see(tk.END)
        
    def _open_notepad(self):
        """Open notepad"""
        subprocess.Popen("notepad.exe")
        self.voice_display.insert(tk.END, "Notepad opened\n")
        self.voice_display.see(tk.END)
        
    def _open_browser(self):
        """Open web browser"""
        webbrowser.open("https://www.google.com")
        self.voice_display.insert(tk.END, "Browser opened\n")
        self.voice_display.see(tk.END)
        
    def _open_paint(self):
        """Open Paint"""
        subprocess.Popen("mspaint.exe")
        self.voice_display.insert(tk.END, "Paint opened\n")
        self.voice_display.see(tk.END)
        
    def _open_excel(self):
        """Open Excel"""
        subprocess.Popen("excel.exe")
        self.voice_display.insert(tk.END, "Excel opened\n")
        self.voice_display.see(tk.END)
        
    def _open_word(self):
        """Open Word"""
        subprocess.Popen("winword.exe")
        self.voice_display.insert(tk.END, "Word opened\n")
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
