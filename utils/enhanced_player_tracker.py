"""
Enhanced player tracking utility with confidence scoring and smart reassignment.
"""

import math
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Set

@dataclass
class TrackingState:
    """Represents the current state of player tracking."""
    original_id: int
    current_id: int
    confidence: float
    lost_frames: int
    original_lost_frames: int
    last_known_position: Optional[Tuple[float, float]]
    position_history: deque
    velocity_estimate: Optional[Tuple[float, float]]
    id_history: Set
    is_temporary_assignment: bool = False
    
class EnhancedPlayerTracker:
    """
    Enhanced player tracking with confidence scoring and smart reassignment.
    """
    
    def __init__(self, max_lost_frames=30, confidence_threshold=0.7, 
                 max_reassignment_distance=150, history_length=10):
        self.max_lost_frames = max_lost_frames
        self.confidence_threshold = confidence_threshold
        self.max_reassignment_distance = max_reassignment_distance
        self.history_length = history_length
        self.tracking_state: Optional[TrackingState] = None
        self.reassignment_candidates = []
        
    def initialize_tracking(self, player_id: int, position: Tuple[float, float]) -> None:
        """Initialize tracking for a specific player."""
        self.tracking_state = TrackingState(
            original_id=player_id,
            current_id=player_id,
            confidence=1.0,
            lost_frames=0,
            original_lost_frames=0,
            last_known_position=position,
            position_history=deque([position], maxlen=self.history_length),
            velocity_estimate=None,
            is_temporary_assignment=False,
            id_history={player_id},
        )
        
    def calculate_player_confidence(self, player_id: int, track_data: dict, 
                                  predicted_position: Optional[Tuple[float, float]] = None) -> float:
        """
        Calculate confidence score for a player being the tracked target.
        
        Factors considered:
        - Distance from predicted position
        - Movement consistency with velocity
        """
        if not self.tracking_state:
            return 0.0
            
        confidence_factors = []
        
        # Factor 1: Spatial proximity
        current_pos = track_data['bbox_center']
        reference_pos = predicted_position or self.tracking_state.last_known_position
        
        if reference_pos:
            distance = math.sqrt((current_pos[0] - reference_pos[0])**2 + 
                               (current_pos[1] - reference_pos[1])**2)
            spatial_confidence = max(0, 1 - (distance / self.max_reassignment_distance))
            confidence_factors.append(spatial_confidence * 0.6)  # 60% weight
        
        # Factor 2: Velocity consistency
        if (self.tracking_state.velocity_estimate and 
            len(self.tracking_state.position_history) >= 2):
            expected_velocity = self.tracking_state.velocity_estimate
            if len(self.tracking_state.position_history) >= 2:
                last_pos = list(self.tracking_state.position_history)[-1]
                velocity_diff = math.sqrt(
                    (current_pos[0] - last_pos[0] - expected_velocity[0])**2 +
                    (current_pos[1] - last_pos[1] - expected_velocity[1])**2
                )
                velocity_confidence = max(0, 1 - (velocity_diff / 100))  # Normalize
                confidence_factors.append(velocity_confidence * 0.4)  # 40% weight
        
        return sum(confidence_factors) if confidence_factors else 0.0
    
    def predict_next_position(self) -> Optional[Tuple[float, float]]:
        """Predict where the tracked player should be based on velocity."""
        if (not self.tracking_state or 
            len(self.tracking_state.position_history) < 2):
            return self.tracking_state.last_known_position if self.tracking_state else None
            
        positions = list(self.tracking_state.position_history)
        
        # Calculate average velocity over recent frames
        velocities = []
        for i in range(1, min(5, len(positions))):  # Use last 4 movements
            vx = positions[i][0] - positions[i-1][0]
            vy = positions[i][1] - positions[i-1][1]
            velocities.append((vx, vy))
        
        if velocities:
            avg_vx = sum(v[0] for v in velocities) / len(velocities)
            avg_vy = sum(v[1] for v in velocities) / len(velocities)
            self.tracking_state.velocity_estimate = (avg_vx, avg_vy)
            
            last_pos = positions[-1]
            predicted_pos = (last_pos[0] + avg_vx, last_pos[1] + avg_vy)
            return predicted_pos
        
        return self.tracking_state.last_known_position
    
    def find_best_reassignment_candidate(self, player_track: Dict) -> Tuple[Optional[int], float]:
        """
        Find the best candidate for reassignment based on confidence scores.
        """
        if not self.tracking_state:
            return None, 0.0
            
        predicted_position = self.predict_next_position()
        best_candidate = None
        best_confidence = 0.0
        
        for player_id, track_data in player_track.items():
            confidence = self.calculate_player_confidence(
                player_id, track_data, predicted_position
            )
            
            if confidence > best_confidence and confidence > self.confidence_threshold:
                best_confidence = confidence
                best_candidate = player_id
        
        return best_candidate, best_confidence
    
    def update_tracking(self, player_track: Dict) -> Tuple[Optional[int], str, bool]:
        """
        Update tracking state and return current tracked ID, status, and whether user input is needed.
        
        Returns:
            Tuple of (tracked_player_id, status_message, needs_user_input)
        """
        if not self.tracking_state:
            return None, "No tracking initialized", False
            
        # Check if original player is back
        if (self.tracking_state.original_id in player_track and 
            self.tracking_state.current_id != self.tracking_state.original_id):
            original_confidence = self.calculate_player_confidence(
                self.tracking_state.original_id, 
                player_track[self.tracking_state.original_id],
                self.predict_next_position()
            )
            self.tracking_state.current_id = self.tracking_state.original_id
            self.tracking_state.is_temporary_assignment = False
            self.tracking_state.lost_frames = 0
            self.tracking_state.original_lost_frames = 0
            self.tracking_state.confidence = original_confidence
            self._update_position_history(player_track[self.tracking_state.current_id]['bbox_center'])
            return (self.tracking_state.current_id, 
                    f"Original player {self.tracking_state.original_id} returned", False)
        
        # Check if current tracked player is still visible
        if self.tracking_state.current_id in player_track:
            current_confidence = self.calculate_player_confidence(
                self.tracking_state.current_id,
                player_track[self.tracking_state.current_id],
                self.predict_next_position()
            )
            self.tracking_state.lost_frames = 0
            self.tracking_state.confidence = current_confidence
            self._update_position_history(player_track[self.tracking_state.current_id]['bbox_center'])
            
            # If we're tracking a temporary assignment, keep counting original lost frames
            if self.tracking_state.is_temporary_assignment:
                self.tracking_state.original_lost_frames += 1
                
                # Check if original has been lost too long - need user confirmation
                if self.tracking_state.original_lost_frames > self.max_lost_frames:
                    return (self.tracking_state.current_id, 
                           f"Confirm temporary assignment: Keep tracking player {self.tracking_state.current_id} as permanent replacement for {self.tracking_state.original_id}?",
                           True)
                else:
                    status = f"Tracking temporary substitute (ID {self.tracking_state.current_id}) - original lost for {self.tracking_state.original_lost_frames}/{self.max_lost_frames} frames"
                    return self.tracking_state.current_id, status, False
            else:
                return self.tracking_state.current_id, "Tracking normally", False
        
        self.tracking_state.lost_frames += 1
        if self.tracking_state.is_temporary_assignment:
            self.tracking_state.original_lost_frames += 1
        else:
            self.tracking_state.original_lost_frames = self.tracking_state.lost_frames
        
        if self.tracking_state.lost_frames <= self.max_lost_frames:
            candidate_id, candidate_confidence = self.find_best_reassignment_candidate(player_track)
            
            if candidate_id:
                old_id = self.tracking_state.current_id
                self.tracking_state.current_id = candidate_id
                self.tracking_state.is_temporary_assignment = (candidate_id != self.tracking_state.original_id)
                self.tracking_state.confidence = candidate_confidence
                self.tracking_state.lost_frames = 0
                self._update_position_history(player_track[candidate_id]['bbox_center'])
                
                return (candidate_id, 
                       f"Reassigned from {old_id} to {candidate_id} (confidence: {candidate_confidence:.2f})",
                       False)
        
        if self.tracking_state.lost_frames > self.max_lost_frames:
            return (None, 
                   f"Player lost for {self.tracking_state.lost_frames} frames. Need user selection.",
                   True)
        else:
            return (None, 
                   f"Player lost for {self.tracking_state.lost_frames}/{self.max_lost_frames} frames",
                   False)
    
    def _update_position_history(self, position: Tuple[float, float]):
        """Update position history and velocity estimates."""
        if self.tracking_state:
            self.tracking_state.position_history.append(position)
            self.tracking_state.last_known_position = position
    
    def get_reassignment_suggestions(self, player_track: Dict, top_n: int = 3) -> List[Tuple[int, float]]:
        """Get top N reassignment suggestions with confidence scores."""
        suggestions = []
        predicted_position = self.predict_next_position()
        
        for player_id, track_data in player_track.items():
            confidence = self.calculate_player_confidence(
                player_id, track_data, predicted_position
            )
            suggestions.append((player_id, confidence))
        
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:top_n]
    
    def confirm_reassignment(self, player_id: int, player_track: Dict):
        """Confirm a reassignment choice from the user."""
        if self.tracking_state and player_id in player_track:
            self.tracking_state.current_id = player_id
            self.tracking_state.is_temporary_assignment = (player_id != self.tracking_state.original_id)
            self.tracking_state.lost_frames = 0
            self.tracking_state.original_lost_frames = 0
            self.tracking_state.confidence = 1.0  # User confirmed
            self._update_position_history(player_track[player_id]['bbox_center'])
    
    def confirm_temporary_as_permanent(self):
        """Confirm the current temporary assignment as the new permanent target."""
        if self.tracking_state and self.tracking_state.is_temporary_assignment:
            self.tracking_state.original_id = self.tracking_state.current_id
            self.tracking_state.is_temporary_assignment = False
            self.tracking_state.original_lost_frames = 0
            self.tracking_state.confidence = 1.0
            self.tracking_state.id_history.add(self.tracking_state.original_id)
            return True
        return False
    
    def deny_temporary_assignment(self):
        """Deny the temporary assignment and prepare for new user selection."""
        if self.tracking_state:
            self.tracking_state.current_id = self.tracking_state.original_id
            self.tracking_state.is_temporary_assignment = False
            self.tracking_state.lost_frames = self.max_lost_frames + 1
            self.tracking_state.original_lost_frames = self.max_lost_frames + 1
            return True
        return False
