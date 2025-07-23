from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from torchreid.reid.utils import FeatureExtractor
import torch
import cv2
from collections import defaultdict, deque

class GlobalIDManager:
    """
    A class to manage consistent global player IDs across frames using cosine similarity
    of appearance embeddings.

    Attributes:
        threshold (float): Cosine similarity threshold to determine ID match.
        max_ids (int): Maximum number of global IDs allowed.
        next_id (int): Counter to assign the next available global ID.
        embeddings (List[np.ndarray]): List of known normalized feature embeddings.
        global_ids (List[int]): List of global IDs corresponding to stored embeddings.
        extractor (FeatureExtractor): Model used to extract embeddings from player images.
        track_buffers (defaultdict): Buffer storing recent embeddings for smoothing.
    """

    def __init__(self, threshold=0.7, max_ids=100):
        """
        Initializes the GlobalIDManager with a similarity threshold and max number of IDs.

        Args:
            threshold (float): Similarity threshold to match embeddings.
            max_ids (int): Maximum number of global IDs to assign.
        """
        self.embeddings = []
        self.global_ids = []
        self.threshold = threshold
        self.max_ids = max_ids
        self.next_id = 1
        self.extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.track_buffers = defaultdict(lambda: deque(maxlen=5))

    def get_global_id(self, embedding):
        """
        Retrieves or assigns a global ID for the given embedding based on cosine similarity.

        Args:
            embedding (np.ndarray): Normalized embedding vector for a detected player.

        Returns:
            int: Assigned global ID.
        
        Raises:
            ValueError: If the number of IDs exceeds the allowed maximum.
        """
        if not self.embeddings:
            if self.next_id > self.max_ids:
                raise ValueError("Maximum number of global IDs exceeded.")
            self.embeddings.append(embedding)
            self.global_ids.append(self.next_id)
            self.next_id += 1
            return self.global_ids[-1]

        similarities = cosine_similarity([embedding], self.embeddings)[0]
        max_sim = np.max(similarities)
        best_idx = np.argmax(similarities)

        if max_sim >= self.threshold:
            return self.global_ids[best_idx]
        else:
            if self.next_id > self.max_ids:
                raise ValueError("Maximum number of global IDs exceeded.")
            self.embeddings.append(embedding)
            self.global_ids.append(self.next_id)
            self.next_id += 1
            return self.global_ids[-1]
        
    def add_to_buffer(self, local_id, embedding):
        """
        Stores a new embedding in a fixed-length buffer for a given local track ID.

        Args:
            local_id (int): The local ID assigned to the player in the current frame.
            embedding (np.ndarray): The feature embedding for the player.
        """
        self.track_buffers[local_id].append(embedding)

    def get_smoothed_embedding(self, local_id):
        """
        Returns the average (smoothed) embedding for a given local track ID.

        Args:
            local_id (int): The local ID of the player.

        Returns:
            np.ndarray or None: Averaged embedding if available, else None.
        """
        buf = self.track_buffers[local_id]
        return np.mean(buf, axis=0) if buf else None

    def get_player_embedding(self, image):
        """
        Extracts a normalized appearance embedding for a player from an image.

        Args:
            image (np.ndarray): BGR image of the player (as from OpenCV).

        Returns:
            np.ndarray or None: Normalized embedding vector or None if invalid.
        """
        if image is None or image.size == 0:
            return None

        # Convert from BGR to RGB as torchreid expects RGB
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Extract features using the re-ID model
        features = self.extractor(rgb_img)  # Shape: (1, 2048)
        features = features.squeeze(0).cpu().numpy()

        # Normalize the embedding
        norm = np.linalg.norm(features)
        if norm == 0:
            return None
        return features / norm
