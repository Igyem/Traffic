import cv2
from ultralytics import YOLO
from utils.tracker import initialize_tracker
from utils.preprocess import draw_annotations, process_roi

# Initialize YOLO model
model = YOLO("models/yolov5s.pt")

def count_vehicles(video_path, roi_line):
    """Detect, track, and count vehicles in a video."""
    vehicle_count = 0
    tracker = initialize_tracker()
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model(frame)
        detections = [
            ((int(box[0]), int(box[1]), int(box[2]) - int(box[0]), int(box[3]) - int(box[1])), 
             float(conf), 
             int(cls))
            for box, conf, cls in zip(results[0].boxes.xyxy.cpu().numpy(), 
                                      results[0].boxes.conf.cpu().numpy(), 
                                      results[0].boxes.cls.cpu().numpy())
        ]

        # Update tracker
        tracks = tracker.update_tracks(detections, frame=frame)

        # Check ROI and count vehicles
        vehicle_count = process_roi(tracks, roi_line, vehicle_count)

        # Annotate frame (optional for debugging)
        draw_annotations(frame, tracks)

    cap.release()
    return vehicle_count
