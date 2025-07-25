"""
Base streaming tracker that processes frames one at a time.
"""

class StreamingTracker:
    """
    Base class for streaming trackers that maintain state across frames.
    """
    
    def __init__(self):
        self.frame_count = 0
        self.state = {}
        
    def reset(self):
        """Reset tracker state for new video."""
        self.frame_count = 0
        self.state = {}
        
    def process_frame(self, frame):
        """
        Process a single frame. Should be implemented by subclasses.
        
        Args:
            frame: Video frame as numpy array.
            
        Returns:
            dict: Tracking results for this frame.
        """
        raise NotImplementedError("Subclasses must implement process_frame")
        
    def finalize(self):
        """
        Perform any final processing after all frames. 
        Can be overridden by subclasses.
        
        Returns:
            Any final results or None.
        """
        return None
