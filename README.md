## Inspiration
We wanted to build an AI-powered desktop assistant that feels natural and intuitive. Voice assistants exist, but few combine gesture tracking and automation. We wanted a tool that understands you and reacts instantly, using your voice or your movements.

## What it does
JarvAIs lets you control your computer with your finger and your voice.
It uses CoreML and OpenCV to track finger movement in real time and move the cursor.
You can say commands like:
- “Jarvis Calculator” to open the calculator
- “Jarvis Notepad” to open text editing
- “Jarvis Lock” or “Jarvis Shutdown” for system control

JarvAIs creates a hands-free, AI-driven interface between you and your machine.

## How we built it
- CoreML for model inference and gesture recognition
- OpenCV for real-time hand and finger tracking
- Python and Swift for cross-platform logic and UI
- Integrated voice commands using lightweight speech recognition
- Mapped system automation commands to natural language triggers

AI + CV + Voice => Human-Computer Symbiosis

## Challenges we ran into
- Tracking finger movements accurately under changing light
- Calibrating OpenCV contours to detect fingertips without false positives
- Linking system-level automation to natural voice input
- Managing latency between gesture and cursor movement

## Accomplishments that we're proud of
- Built a functional system that tracks and moves your cursor using gestures
- Integrated a working AI bot that responds to commands
- Made system control intuitive without a keyboard or mouse
- Achieved smooth performance and stable gesture tracking

## What we learned
- Real-time vision models require strong optimization
- Voice command mapping improves usability
- Combining CV and NLP systems creates better user control
- Fine-tuning CoreML models improves accuracy significantly

## What's next for JarvAIs
- Add more system and app integrations
- Improve hand-tracking precision with custom-trained models
- Create a minimal on-screen interface for feedback
- Add support for multi-finger gestures
- Deploy as a macOS and Windows desktop app
