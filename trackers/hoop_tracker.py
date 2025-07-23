from ultralytics import YOLO
import supervision as sv
import sys
sys.path.append('../')
from utils import save_stub, read_stub
import config

class HoopTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames):
        """
        Detect hoops in a sequence of frames using batch processing.

        Args:
            frames (list): List of video frames to process.

        Returns:
            list: YOLO detection results for each frame.
        """
        batch_size = config.batch_size
        detections = []

        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]
            detections_batch = self.model.predict(batch_frames)
            detections += detections_batch

        return detections
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Track the hoop(s) accross frames using supervision

        Args:
            frames (list): List of video frames to process.
            read_from_stub (bool): whether to read in cached results
            stub_path (str): path to the cached file
        """
        tracks = read_stub(read_from_stub, stub_path)
        if tracks is not None:
            if len(tracks) == len(frames):
                return tracks 
            
        detections = self.detect_frames(frames)
        tracks = []

        for i, detection in enumerate(detections):
            frame_tracks = {}
            cls_names = detection.names
            cls_names_inv = {v : k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)
            hoop_id = 0

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                confidence = frame_detection[2]
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv[config.hoop_label]:
                    if hoop_id == 2:
                        print("======MORE THAN 2 HOOPS IDENTIFIED======")
                        min_conf_id = min(frame_tracks, key=lambda k: frame_tracks[k]["confidence"])
                        if confidence > frame_tracks[min_conf_id]["confidence"]:
                            frame_tracks[min_conf_id] = {"bbox": bbox, "confidence": confidence}
                    else:
                        frame_tracks[hoop_id] = {"bbox": bbox, "confidence": confidence}
                        hoop_id += 1

            tracks.append(frame_tracks)
        
        save_stub(stub_path, tracks)

        return tracks

    