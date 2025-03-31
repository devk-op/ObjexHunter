# ObjexHunter

ObjexHunter is a project designed to perform object detection and send notifications using Twilio. It includes scripts for desktop-based object detection and Twilio integration for sending messages.

## Features
- Object detection using YOLO models.
- Twilio integration for sending SMS notifications.
- Easy-to-use Python scripts.

## Prerequisites
- Python 3.8 or higher
- Required Python libraries (listed in `requirements.txt`)
- Twilio account for SMS functionality

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/devk-op/ObjexHunter.git
   cd ObjexHunter
2. python3 -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
3. pip install -r requirements.txt

File Structure
ObjexHunter/
├── desktop.py              # Script for object detection without SMS
├── twilio.py               # Script for object detection with Twilio SMS
├── requirements.txt        # Python dependencies
├── [ReadMe.txt](http://_vscodecontentref_/1)              # Original instructions
└── README.md               # Project documentation