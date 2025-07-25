from utils import GlobalIDManager
from ultralytics import YOLO
from collections import defaultdict
import supervision as sv
import sys
sys.path.append('../')
from utils import read_stub, save_stub
import config

class PlayerTracker:
    """
    A class that handles player detection and tracking using YOLO and ByteTrack.

    This class combines YOLO object detection with ByteTrack tracking to maintain consistent
    player identities across frames while processing detections in batches.
    """
    def __init__(self, model_path):
        """
        Initialize the PlayerTracker with YOLO model and ByteTrack tracker.

        Args:
            model_path (str): Path to the YOLO model weights.
        """
        self.model = YOLO(model_path) 
        self.tracker = sv.ByteTrack()
        # self.global_id_manager = GlobalIDManager()
        self.track_embeddings = defaultdict(list)
        self.track_bboxes = defaultdict(dict)

    def detect_frames(self, frames):
        """
        Detect players in a sequence of frames using batch processing.

        Args:
            frames (list): List of video frames to process.

        Returns:
            list: YOLO detection results for each frame.
        """
        batch_size = config.batch_size
        detections = []

        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]
            detections_batch = self.model.predict(batch_frames, classes=[0])
            detections += detections_batch

        return detections
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Track players accross frames using supervision

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

        print("===Assigning tracks and global ids=====")
        for frame_num, detection in enumerate(detections):
            frame_tracks = {}
            cls_names = detection.names
            cls_names_inv = {v : k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                local_id = frame_detection[4]

                if cls_id == cls_names_inv[config.player_label]:
                    # x1, y1, x2, y2 = map(int, bbox)
                    # player_crop = frames[frame_num][y1:y2, x1:x2]

                    # embedding = self.global_id_manager.get_player_embedding(player_crop)
                    # if embedding is None:
                    #     continuel

                    # self.track_embeddings[local_id].append(embedding)
                    # self.track_bboxes[frame_num][local_id] = bbox

                    # smoothed = self.global_id_manager.get_smoothed_embedding(local_id)
                    # if smoothed is None:
                    #     continue

                    # global_id = self.global_id_manager.get_global_id(smoothed)
                    frame_tracks[local_id] = {"bbox" : bbox, "local_id": local_id}

            tracks.append(frame_tracks)

            # for local_id, box in enumerate(detection.boxes):
            #     if int(box.cls) == cls_names_inv[config.player_label]:
            #         bbox = box.xyxy.cpu().numpy().flatten().tolist()  # [x1, y1, x2, y2]
            #         frame_tracks[local_id] = {'bbox': bbox, "local_id": local_id}
            # tracks.append(frame_tracks)
        
        save_stub(stub_path, tracks)

        return tracks
