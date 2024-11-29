import cv2

def draw_annotations(frame, tracks):
    """Annotate frame with bounding boxes and IDs."""
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        bbox = track.to_tlbr()
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, bbox)

        # Draw bounding box and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def process_roi(tracks, roi_line, vehicle_count):
    """Count vehicles crossing an ROI."""
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        bbox = track.to_tlbr()
        x1, y1, x2, y2 = map(int, bbox)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        if roi_line[0][1] < center[1] < roi_line[1][1]:  # Example ROI check
            vehicle_count += 1
    return vehicle_count
