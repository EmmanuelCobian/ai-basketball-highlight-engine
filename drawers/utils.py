"""
A utility module providing functions for drawing shapes on video frames.

This module includes functions to draw triangles and ellipses on frames, which can be used
to represent various annotations such as player positions or ball locations in sports analysis.
"""
import cv2 
import numpy as np
import sys 
sys.path.append('../')
from utils import get_bbox_center, get_bbox_width

def draw_traingle(frame, bbox, color):
    """
    Draws a filled triangle on the given frame at the specified bounding box location.

    Args:
        frame (numpy.ndarray): The frame on which to draw the triangle.
        bbox (tuple): A tuple representing the bounding box (x, y, width, height).
        color (tuple): The color of the triangle in BGR format.

    Returns:
        numpy.ndarray: The frame with the triangle drawn on it.
    """
    y = int(bbox[1])
    x, _ = map(int, get_bbox_center(bbox))

    triangle_points = np.array([
        [x,y],
        [x - 10, y - 20],
        [x + 10, y - 20],
    ])
    cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

    return frame

def draw_box(frame, bbox, color, label=None):
    """
    Draws a rectangle box around an obect

    Args:
        frame (numpy.ndarray): The frame on which to draw
        bbox (tuple): A tuple representing the bounding box (x1, y1, x2, y2)
        color (tuple): the color of the box in RGB format
        label (str, optional): the text label to display

    Returns:
        numpy.ndarray: the frame with the box drawn on it
    """
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(
        frame,
        pt1=(x1, y1),
        pt2=(x2, y2),
        color=color,
        thickness=3,
    )

    return frame

def draw_ellipse(frame, bbox, color, label=None):
    """
    Draws an ellipse and an optional rectangle with a label (e.g., "G:5 / L:3")
    on the given frame at the specified bounding box location.

    Args:
        frame (numpy.ndarray): The frame on which to draw the ellipse.
        bbox (tuple): A tuple representing the bounding box (x1, y1, x2, y2).
        color (tuple): The color of the ellipse in RGB format.
        label (str, optional): The text label to display. Defaults to None.

    Returns:
        numpy.ndarray: The frame with the ellipse and optional label drawn on it.
    """
    y2 = int(bbox[3])
    x_center, _ = map(int, get_bbox_center(bbox))
    width = get_bbox_width(bbox)

    # Draw ellipse under player
    cv2.ellipse(
        frame,
        center=(x_center, y2),
        axes=(int(width), int(0.35 * width)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=color,
        thickness=2,
        lineType=cv2.LINE_4
    )

    if label is not None:
        # Measure text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)

        # Compute box around label
        padding = 10
        box_width = text_width + padding
        box_height = text_height + padding

        x1_rect = x_center - box_width // 2
        x2_rect = x_center + box_width // 2
        y1_rect = y2 + 15
        y2_rect = y1_rect + box_height

        # Draw filled rectangle
        cv2.rectangle(frame, (x1_rect, y1_rect), (x2_rect, y2_rect), color, cv2.FILLED)

        # Put the label text
        text_x = x_center - text_width // 2
        text_y = y1_rect + box_height - 7
        cv2.putText(frame, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

    return frame
