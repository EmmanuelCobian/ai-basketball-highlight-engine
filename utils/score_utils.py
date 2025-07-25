import math
import numpy as np
import torch
from .bbox_utils import get_bbox_center, get_bbox_height, get_bbox_width

def get_device():
    """Automatically select devices -> mps（Mac） -> cpu"""
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return device

def get_closest_hoop(ball_cx, ball_cy, hoop_bboxes):
    min_dist = float('inf')
    closest_hoop_id = None
    for hoop_id, data in hoop_bboxes.items():
        hoop_cx, hoop_cy = get_bbox_center(data['bbox'])
        dist = (ball_cx - hoop_cx) ** 2 + (ball_cy - hoop_cy) ** 2
        if dist < min_dist:
            min_dist = dist
            closest_hoop_id = hoop_id
    return closest_hoop_id


def score(ball_pos, hoop_pos, frame_num):
    """
    Determines whether a basketball shot is likely to score based on ball and hoop positions over a sequence of frames.

    Args:
        ball_pos (dict): Dictionary mapping frame numbers to ball position data.
            Example: ball_pos[frame_num] = {1: {'bbox': [...]}}
        hoop_pos (dict): Dictionary mapping frame numbers to hoop position data.
            Example: hoop_pos[frame_num] = {0: {'bbox': [...]}, 1: {'bbox': [...]}}
        frame_num (int): The current frame number to analyze.

    Returns:
        bool: True if the shot is likely to score (ball passes through the rim area), False otherwise.

    Notes:
        - Uses the ball's trajectory over the last 30 frames (or fewer if near the start).
        - Fits a linear model to the ball's path to predict where it will cross the rim height.
        - Considers a margin around the rim to account for near-misses and rebounds.
        - ball_pos[frame_num] = {1 : {'bbox': [...]}}
        - hoop_pos[frame_num] = {0 : {'bbox': [...]}, 1 : {'bbox': [...]}}
    """
    ball_cx, ball_cy = get_bbox_center(ball_pos[frame_num][1]['bbox'])

    clossest_hoop_id = get_closest_hoop(ball_cx, ball_cy, hoop_pos[frame_num])
    hoop_bbox = hoop_pos[frame_num][clossest_hoop_id]['bbox']
    hoop_cx, hoop_cy = get_bbox_center(hoop_bbox)
    hoop_height = get_bbox_height(hoop_bbox)
    hoop_width = get_bbox_width(hoop_bbox)

    x, y = [], []
    rim_height = hoop_cy - 0.5 * hoop_height

    for i in reversed(range(max(0, frame_num - 30), frame_num + 1)):
        ball_bbox = ball_pos[i][1]['bbox']
        cx, cy = get_bbox_center(ball_bbox)
        if cy < rim_height:
            x.append(cx)
            y.append(cy)
            if i + 1 < frame_num + 1:
                next_ball_bbox = ball_pos[i + 1][1]['bbox']
                n_cx, n_cy = get_bbox_center(next_ball_bbox)
                x.append(n_cx)
                y.append(n_cy)
            break

    if len(x) > 1:
        m, b = np.polyfit(x, y, 1)
        predicted_x = ((hoop_cy - 0.5 * hoop_height) - b) / m
        rim_margin = 0.25
        rim_x1 = hoop_cx - rim_margin * hoop_width
        rim_x2 = hoop_cx + rim_margin * hoop_width

        if rim_x1 < predicted_x < rim_x2:
            return True
        hoop_rebound_zone = 0.2 * hoop_width
        if rim_x1 - hoop_rebound_zone < predicted_x < rim_x2 + hoop_rebound_zone:
            return True

    return False


# Detects if the ball is below the net - used to detect shot attempts
def detect_down(ball_pos, hoop_pos):
    # ball_pos = {1 : {'bbox': [...]}}
    # hoop_pos = {0 : {'bbox': [...]}, 1 : {'bbox': [...]}}
    ball_bbox = ball_pos[1]['bbox']
    ball_cx, ball_cy = get_bbox_center(ball_bbox)

    clossest_hoop_id = get_closest_hoop(ball_cx, ball_cy, hoop_pos)
    hoop_bbox = hoop_pos[clossest_hoop_id]['bbox']
    _, hoop_cy = get_bbox_center(hoop_bbox)
    hoop_height = get_bbox_height(hoop_bbox)

    y = hoop_cy + 0.5 * hoop_height
    if ball_cy > y:
        return True
        
    return False


def detect_up(ball_pos, hoop_pos):
    # ball_pos = {1 : {'bbox': [...]}}
    # hoop_pos = {0 : {'bbox': [...]}, 1 : {'bbox': [...]}}
    ball_bbox = ball_pos[1]['bbox']
    ball_cx, ball_cy = get_bbox_center(ball_bbox)

    clossest_hoop_id = get_closest_hoop(ball_cx, ball_cy, hoop_pos)
    hoop_bbox = hoop_pos[clossest_hoop_id]['bbox']
    hoop_cx, hoop_cy = get_bbox_center(hoop_bbox)
    hoop_height = get_bbox_height(hoop_bbox)
    hoop_width = get_bbox_width(hoop_bbox)

    ex1 = hoop_cx - 4 * hoop_width
    ex2 = hoop_cx + 4 * hoop_width
    ey1 = hoop_cy - 2 * hoop_height
    ey2 = hoop_cy

    if ex1 < ball_cx < ex2 and ey1 < ball_cy < ey2:
        return True
    return False
