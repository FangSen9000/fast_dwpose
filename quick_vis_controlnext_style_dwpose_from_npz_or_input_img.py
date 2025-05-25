import torch
from PIL import Image
import numpy as np
import json

from easy_dwpose import DWposeDetector
from easy_dwpose.draw.controlnext import draw_pose, process_pose_data 

#####---------Setup init
# You can use a different GPU, e.g. "cuda:1"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
detector = DWposeDetector(device=device)
input_image = Image.open("assets/pose.png").convert("RGB")



#####---------Custom ControlNext drawing style
# Get pose data for custom drawing
#pose_data = detector(input_image, draw_pose=False)
# or
pose_data = dict(np.load('assets/WLASL_01214/results_dwpose/npz/00000011.npz'))

# Get image dimensions
#width, height = input_image.size
# or
width, height = 480, 480  #When I handle it, I default to a square with the largest side length. The size of the WLASL is 480*480.

# Process the pose data for custom drawing
processed_pred = process_pose_data(pose_data, height, width)

# Draw pose using custom ControlNext style
vis_img = draw_pose(
    pose=processed_pred,
    H=height,
    W=width,
    include_body=True,
    include_hand=True,
    include_face=True
)

# Convert to PIL Image and save (vis_img is in CHW format)
custom_skeleton = Image.fromarray(vis_img.transpose(1, 2, 0))
custom_skeleton.save("skeleton_controlnext.png")