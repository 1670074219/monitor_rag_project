# Monitor Page

This folder contains a simple web application to monitor RTSP camera streams.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the server:
   ```bash
   python app.py
   ```

3. Open your browser and navigate to:
   http://localhost:5000

## Configuration

The application reads camera configurations from `../video_process/camera_config.json`.
It expects a JSON structure like:
```json
{
    "camera_config": [
        {
            "camera_id": "camera1",
            "camera_url": "rtsp://..."
        },
        ...
    ]
}
```
