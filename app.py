from flask import Flask, request, jsonify, render_template, Response
import os
from vehicle_counter import count_vehicles
from utils.tracker import initialize_tracker
from utils.preprocess import draw_annotations
from ultralytics import YOLO
import cv2


model = YOLO("models/yolov5su.pt")


# Initialize Flask app
app = Flask(__name__)

# Directory for uploaded videos
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Route: Homepage
@app.route("/")
def index():
    return render_template("index.html")

# Route: Upload Video and Process
@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save uploaded video
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Define ROI for counting (adjust as needed)
    roi_line = [(300, 400), (700, 400)]

    # Process the video and count vehicles
    try:
        vehicle_count = count_vehicles(filepath, roi_line)
        return jsonify({"vehicle_count": vehicle_count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route: Live Video Feed
@app.route("/video_feed")
def video_feed():
    """Stream live video feed."""
    return Response(process_live_feed(), mimetype="multipart/x-mixed-replace; boundary=frame")

def process_live_feed():
    """Capture and process frames from a live video feed."""
    cap = cv2.VideoCapture(0)  # 0 for webcam, or replace with RTSP URL
    tracker = initialize_tracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO and Deep SORT tracking
        results = model(frame)
        detections = [
            ((int(box[0]), int(box[1]), int(box[2]) - int(box[0]), int(box[3]) - int(box[1])), 
             float(conf), 
             int(cls))
            for box, conf, cls in zip(results[0].boxes.xyxy.cpu().numpy(), 
                                      results[0].boxes.conf.cpu().numpy(), 
                                      results[0].boxes.cls.cpu().numpy())
        ]
        tracks = tracker.update_tracks(detections, frame=frame)
        annotated_frame = draw_annotations(frame, tracks)

        # Encode frame as JPEG
        _, buffer = cv2.imencode(".jpg", annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()

# Main entry point
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
