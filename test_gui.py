#!/usr/bin/env python3
"""
Test script for jarvAIs Western Command Center GUI
This script tests the GUI without camera functionality
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from jarvAIs import jarvAIsGUI
    import tkinter as tk
    
    print("🤠 Testing jarvAIs Western Command Center GUI...")
    
    # Create a test version that doesn't start camera
    class TestGUI(jarvAIsGUI):
        def __init__(self):
            # Initialize without camera
            import tkinter as tk
            from PIL import Image, ImageTk
            import subprocess
            import webbrowser
            from datetime import datetime
            
            self.root = tk.Tk()
            self.root.title("🤠 jarvAIs Western Command Center - TEST MODE")
            self.root.geometry("1600x1000")
            
            # Desert Night Theme Colors
            self.colors = {
                'deep_navy': '#0b1b3b',
                'sand_orange': '#c8692a', 
                'gold': '#f6c05b',
                'dark_bg': '#0a0a1a',
                'card_bg': '#1a1a3a',
                'accent': '#ff8c00'
            }
            
            self.root.configure(bg=self.colors['deep_navy'])
            self.root.state('zoomed')
            
            # Mock configuration
            class MockConfig:
                camera_width = 640
                camera_height = 480
                camera_fps = 60
            
            self.config = MockConfig()
            
            # Mock other attributes
            self.cap = None
            self.recognizer = None
            self.running = False
            self.jarvis = None
            self.current_page = 0
            self.pages = ["main", "gestures", "voice", "tools", "settings"]
            
            # Create GUI
            self.create_widgets()
            
        def start_camera(self):
            """Mock camera start"""
            print("📹 Camera test mode - no actual camera started")
            
        def start_gesture_control(self):
            """Mock gesture control"""
            print("🚀 Gesture control test mode activated")
            
        def stop_gesture_control(self):
            """Mock gesture control stop"""
            print("⛔ Gesture control test mode stopped")
            
        def toggle_jarvis(self):
            """Mock JARVIS toggle"""
            print("🎤 JARVIS test mode toggled")
            
        def send_chat_message(self, event=None):
            """Mock chat message"""
            message = self.chat_input.get().strip()
            if message:
                print(f"💬 Test chat message: {message}")
                self.chat_input.delete(0, tk.END)
    
    # Run the test
    print("🌟 Starting GUI test...")
    app = TestGUI()
    print("✅ GUI created successfully!")
    print("🎯 GUI should be visible now - test the interface")
    print("❌ Press Ctrl+C to close")
    
    app.run()
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure all dependencies are installed:")
    print("   pip install opencv-python mediapipe numpy pyautogui autopy SpeechRecognition Pillow")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
