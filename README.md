# ARIA: Autonomous Robotic Intelligence Arm

ARIA is a Raspberry Pi 5-powered robotic hand system that uses local Machine Learning (Gemma 4 via Ollama and OpenAI Whisper) to understand and execute natural language voice commands.

## 🏗 Pipeline Architecture

```text
    🎤 [ USB Mic ]
          │
          ▼
    [ Whisper tiny ] ─── (Speech-to-Text) ──┐
                                            │
                                            ▼
    [ ARIA Executor ] ◄── (Tool Calls) ── [ Gemma 4 ]
          │                               (via Ollama)
          ▼
    [ PCA9685 Driver ] ─── (I2C) ───┐
                                    │
                                    ▼
    [ 6x MG996R Servos ] ◄──── (PWM) ─── [ Robotic Hand ]
```

---

## 🔌 Hardware Setup

### Wiring Table (Pi to PCA9685)
| Raspberry Pi 5 Pin | PCA9685 Pin | Description |
| :--- | :--- | :--- |
| 5V (Pin 2 or 4) | VCC / V+ | Power for Controller & Servos |
| GND (Pin 6) | GND | Ground |
| GPIO 2 (SDA) | SDA | I2C Data |
| GPIO 3 (SCL) | SCL | I2C Clock |

### Servo Channel Mapping
| PCA9685 Channel | Component |
| :--- | :--- |
| Channel 0 | Thumb |
| Channel 1 | Index finger |
| Channel 2 | Middle finger |
| Channel 3 | Ring finger |
| Channel 4 | Pinky finger |
| Channel 5 | Wrist Rotation |

---

## 🚀 Installation

### 1. Prerequisite: Local LLM
ARIA requires [Ollama](https://ollama.com/) to be installed and running.
```bash
# Pull the required model
ollama pull gemma4:e2b
```

### 2. System Dependencies
On Raspberry Pi OS (Linux), you may need `portaudio` for audio capture and `ffmpeg` for Whisper:
```bash
sudo apt update
sudo apt install portaudio19-dev libffi-dev libssl-dev ffmpeg
```

### 3. Python Setup
```bash
git clone https://github.com/yourusername/aria.git
cd aria
pip install -r requirements.txt
```

---

## 🎮 Usage

### 🎤 Voice Control (Normal Mode)
Launch ARIA and speak naturally to the hand.
```bash
python main.py
```

### ⌨️ Text-Only Mode
Control ARIA via keyboard input (no microphone required).
```bash
python main.py --text
```

### 🤖 Demo Sequence
Run a pre-programmed showcase of gestures.
```bash
python main.py --demo
```

### Example Commands
- *"ARIA, give me a peace sign."*
- *"Close your hand into a fist."*
- *"Pinch your thumb and index finger together."*
- *"Wave hello to the crowd!"*
- *"Rotate your wrist 45 degrees to the left."*

---

## 🏅 Hack Club Badge Justification
ARIA satisfies the following technical requirements:
- **Motors**: Controls 6 high-torque MG996R servos.
- **I2C**: Communicates with the PCA9685 PWM driver over the I2C bus.
- **Machine Learning**: Integrates two local ML models (Whisper for STT and Gemma 4 for natural language intent parsing).

---

## 🛡 License
MIT License. Created for the ARIA project.
