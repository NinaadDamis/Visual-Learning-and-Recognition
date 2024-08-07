import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


# TODO: given bounding boxes and corresponding scores, perform non max suppression
def nms(bounding_boxes, confidence_score, threshold=0.05):
    """
    bounding boxes of shape     Nx4
    confidence scores of shape  N
    threshold: confidence threshold for boxes to be considered

    return: list of bounding boxes and scores
    """
    bounding_boxes   = bounding_boxes.cpu().detach().numpy()
    confidence_score = confidence_score.cpu().detach().numpy()
    # Get indexes with score > threshold. rest can be ignored.
    indexes = confidence_score>threshold
    confidence_score = confidence_score[indexes]
    bounding_boxes = bounding_boxes[indexes,:]
    # Get score indexes in sorted order. nms boxes should be placed in descending orfer of scores.
    score_indexes = np.argsort(confidence_score)[::-1]
    # Sorted and theshold > 0.05
    sorted_scores = confidence_score[score_indexes]
    sorted_boxes = bounding_boxes[score_indexes]

    boxes = []
    scores = []
    if len(sorted_boxes) > 0:
        # First element always has to be added to returned list as it has highest score.
        boxes.append(sorted_boxes[0])
        scores.append(sorted_scores[0])
        # remove element from sorted list
        # print(len(sorted_scores))
        #sorted_scores.pop(0)
        # print(len(sorted_scores))
        sorted_scores = sorted_scores[1:]
        sorted_boxes = sorted_boxes[1:]

    while len(sorted_boxes)>0:
        temp_score = sorted_scores[0]
        temp_box = sorted_boxes[0]
        # Check the current box with each box in list to be returned
        for box in boxes:
            # If above threshold, then not added to box list. Else Added.
            if iou(temp_box, box) >= 0.05:
                # We decrease the size of the soted lists. Basically pop() first element.
                sorted_scores = sorted_scores[1:]
                sorted_boxes = sorted_boxes[1:]
            else:
                # Non overlapping, so add to list.
                scores.append(temp_score)
                boxes.append(temp_box)
                # sorted_scores = sorted_scores[1:]
                # sorted_boxes = sorted_boxes[1:]
    # Convert to array and return
    return np.array(boxes), np.array(scores)

# TODO: calculate the intersection over union of two boxes
def iou(box1, box2):
    """
    Calculates Intersection over Union for two bounding boxes (xmin, ymin, xmax, ymax)
    returns IoU vallue
    """
    x1 = max(box1[0],box2[0])
    y1 = max(box1[1],box2[1])
    x2 = min(box1[2],box2[2])
    y2 = min(box1[3],box2[3])

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # intersection = max((x2 - x1),0) * max((y2-y1),0)
    intersection = (x2 - x1) *(y2 - y1) # Should be positive
    union = area1 + area2 - intersection
    iou = intersection / union
    return iou


def tensor_to_PIL(image):
    """
    converts a tensor normalized image (imagenet mean & std) into a PIL RGB image
    will not work with batches (if batch size is 1, squeeze before using this)
    """
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255],
    )

    inv_tensor = inv_normalize(image)
    inv_tensor = torch.clamp(inv_tensor, 0, 1)
    original_image = transforms.ToPILImage()(inv_tensor).convert("RGB")

    return original_image


def get_box_data(classes, bbox_coordinates):
    """
    classes : tensor containing class predictions/gt
    bbox_coordinates: tensor containing [[xmin0, ymin0, xmax0, ymax0], [xmin1, ymin1, ...]] (Nx4)

    return list of boxes as expected by the wandb bbox plotter
    """
    box_list = [{
            "position": {
                "minX": bbox_coordinates[i][0],
                "minY": bbox_coordinates[i][1],
                "maxX": bbox_coordinates[i][2],
                "maxY": bbox_coordinates[i][3],
            },
            "class_id": classes[i],
        } for i in range(len(classes))
        ]

    return box_list
