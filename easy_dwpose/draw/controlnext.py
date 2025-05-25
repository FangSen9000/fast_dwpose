import math
import numpy as np
import matplotlib
import cv2
import sys
import os
import _pickle as cPickle
import gzip
import subprocess
import torch
import colorsys
from typing import List, Dict, Any, Optional, Tuple


eps = 0.01

def alpha_blend_color(color, alpha):
    """blend color according to point conf
    """
    return [int(c * alpha) for c in color]

def draw_bodypose(canvas, candidate, subset, score, transparent=False):
    """Draw body pose on canvas
    Args:
        canvas: numpy array canvas to draw on
        candidate: pose candidate
        subset: pose subset
        score: confidence scores
        transparent: whether to use transparent background
    Returns:
        canvas: drawn canvas
    """
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    stickwidth = 4

    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    # Add alpha channel if transparent
    if transparent:
        colors = [color + [255] for color in colors]

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            conf = score[n][np.array(limbSeq[i]) - 1]
            if conf[0] < 0.3 or conf[1] < 0.3:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            if transparent:
                color = colors[i][:-1] + [int(255 * conf[0] * conf[1])]  # Adjust alpha based on confidence
            else:
                color = colors[i]
            cv2.fillConvexPoly(canvas, polygon, color)

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            conf = score[n][i]
            x = int(x * W)
            y = int(y * H)
            if transparent:
                color = colors[i][:-1] + [int(255 * conf)]  # Adjust alpha based on confidence
            else:
                color = colors[i]
            cv2.circle(canvas, (int(x), int(y)), 4, color, thickness=-1)

    return canvas

def draw_handpose(canvas, all_hand_peaks, all_hand_scores, transparent=False):
    """Draw hand pose on canvas"""
    H, W, C = canvas.shape

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10],
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    for peaks, scores in zip(all_hand_peaks, all_hand_scores):
        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            score = scores[e[0]] * scores[e[1]]
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                color = matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0])
                if transparent:
                    color = np.append(color, score)  # Add alpha channel
                else:
                    color = color * score
                cv2.line(canvas, (x1, y1), (x2, y2), color * 255, thickness=2)

        for i, keypoint in enumerate(peaks):
            x, y = keypoint
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                if transparent:
                    color = (0, 0, 0, scores[i])  # Black with alpha
                else:
                    color = (0, 0, int(scores[i] * 255))  # Original color
                cv2.circle(canvas, (x, y), 4, color, thickness=-1)
    return canvas

def draw_facepose(canvas, all_lmks, all_scores, transparent=False):
    """Draw face pose on canvas"""
    H, W, C = canvas.shape
    for lmks, scores in zip(all_lmks, all_scores):
        for lmk, score in zip(lmks, scores):
            x, y = lmk
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                if transparent:
                    color = (255, 255, 255, int(score * 255))  # White with alpha
                else:
                    conf = int(score * 255)
                    color = (conf, conf, conf)  # Original grayscale
                cv2.circle(canvas, (x, y), 3, color, thickness=-1)
    return canvas

def draw_pose(pose, H, W, include_body=True, include_hand=True, include_face=True, ref_w=2160, transparent=False):
    """vis dwpose outputs with optional transparent background

    Args:
        pose (List): DWposeDetector outputs
        H (int): height
        W (int): width
        include_body (bool): whether to draw body keypoints
        include_hand (bool): whether to draw hand keypoints
        include_face (bool): whether to draw face keypoints
        ref_w (int, optional): reference width. Defaults to 2160.
        transparent (bool, optional): whether to use transparent background. Defaults to False.

    Returns:
        np.ndarray: image pixel value in RGBA mode if transparent=True, otherwise RGB mode
    """
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']

    sz = min(H, W)
    sr = (ref_w / sz) if sz != ref_w else 1

    # Create canvas - now with alpha channel if transparent
    if transparent:
        canvas = np.zeros(shape=(int(H*sr), int(W*sr), 4), dtype=np.uint8)
    else:
        canvas = np.zeros(shape=(int(H*sr), int(W*sr), 3), dtype=np.uint8)

    if include_body:
        canvas = draw_bodypose(canvas, candidate, subset, score=bodies['score'], transparent=transparent)

    if include_hand:
        canvas = draw_handpose(canvas, hands, pose['hands_score'], transparent=transparent)

    if include_face:
        canvas = draw_facepose(canvas, faces, pose['faces_score'], transparent=transparent)

    if transparent:
        return cv2.cvtColor(cv2.resize(canvas, (W, H)), cv2.COLOR_BGRA2RGBA).transpose(2, 0, 1)
    else:
        return cv2.cvtColor(cv2.resize(canvas, (W, H)), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)

def process_pose_data(pose_data: Dict[str, Any], height: int, width: int) -> Dict[str, Any]:
    """
    处理姿势数据，保持原始的-1标记，确保只连接有效点，并调整坐标以保持正确比例
    """
    processed_data = {}
    
    # 获取原始数据
    bodies = pose_data['bodies'].copy()
    body_scores = pose_data['body_scores'].reshape(1, -1)
    
    # 计算缩放和偏移
    min_dim = min(height, width)
    offset_x = (width - min_dim) / 2  # 水平居中的偏移量
    
    # 调整坐标，使用较小的维度作为缩放基准，并居中
    adjusted_bodies = bodies.copy()
    # X坐标：先缩放到min_dim，然后加上偏移使其居中
    adjusted_bodies[:, 0] = bodies[:, 0] * min_dim + offset_x
    # Y坐标：直接使用min_dim进行缩放
    adjusted_bodies[:, 1] = bodies[:, 1] * min_dim
    
    # 将调整后的坐标重新归一化到[0,1]范围
    adjusted_bodies[:, 0] /= width
    adjusted_bodies[:, 1] /= height
    
    # 创建subset和scores
    subset = body_scores.copy()
    scores = np.zeros_like(body_scores)
    valid_mask = (body_scores != -1)[0]
    scores[0, valid_mask] = 1.0
    
    processed_data['bodies'] = {
        'candidate': adjusted_bodies,  # 使用调整后的坐标
        'subset': subset,
        'score': scores
    }
    
    # 调整手部坐标
    adjusted_hands = pose_data['hands'].copy()
    for hand in adjusted_hands:
        hand[:, 0] = hand[:, 0] * min_dim + offset_x
        hand[:, 1] = hand[:, 1] * min_dim
        hand[:, 0] /= width
        hand[:, 1] /= height
    
    processed_data['hands'] = adjusted_hands
    processed_data['hands_score'] = pose_data['hands_scores']
    
    # 调整面部坐标
    adjusted_faces = pose_data['faces'].copy()
    for face in adjusted_faces:
        face[:, 0] = face[:, 0] * min_dim + offset_x
        face[:, 1] = face[:, 1] * min_dim
        face[:, 0] /= width
        face[:, 1] /= height
    
    processed_data['faces'] = adjusted_faces
    processed_data['faces_score'] = pose_data['faces_scores']
    
    return processed_data