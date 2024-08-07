from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import argparse
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import numpy as np
from datetime import datetime
import pickle as pkl

# imports
from wsddn import WSDDN
from voc_dataset import *
import wandb
from utils import nms, tensor_to_PIL, iou, get_box_data
from PIL import Image, ImageDraw
from task_1 import AverageMeter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
class_id_to_label = dict(enumerate(CLASS_NAMES))

USE_WANDB = True
if USE_WANDB:
    wandb.init(project="COLAB-VLR-Task2", reinit=True)
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument(
    '--lr',
    default=0.0001,
    type=float,
    help='Learning rate'
)
parser.add_argument(
    '--lr-decay-steps',
    default=150000,
    type=int,
    help='Interval at which the lr is decayed'
)
parser.add_argument(
    '--lr-decay',
    default=0.1,
    type=float,
    help='Decay rate of lr'
)
parser.add_argument(
    '--momentum',
    default=0.9,
    type=float,
    help='Momentum of optimizer'
)
parser.add_argument(
    '--weight-decay',
    default=0.0004,
    type=float,
    help='Weight decay'
)
parser.add_argument(
    '--epochs',
    default=6,
    type=int,
    help='Number of epochs'
)
parser.add_argument(
    '--val-interval',
    default=500,
    type=int,
    help='Interval at which to perform validation'
)
parser.add_argument(
    '--disp-interval',
    default=10,
    type=int,
    help='Interval at which to perform visualization'
)
parser.add_argument(
    '--use-wandb',
    default=False,
    type=bool,
    help='Flag to enable visualization'
)
# ------------

# Set random seed
rand_seed = 1024
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)

# Set output directory
output_dir = "./"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

### Q2.3 
### calculate_map(true_pos,total,pos,ground_truth)
### Inputs - true_pos is an array containing number of true positives for each class (TP)
###          total_pos is array contaning number of predicted positives for each class (TP + FP)
###          ground truth is an array contaning the labels for each class.(TP + FN)
###
### Precision = TP / (TP+FP) , Recall = TP / (TP+FN)
### We obtain AP of each class by performing an element wise multiplication of precision and recall
### mAP = Mean of AP of all classes
def calculate_map(true_pos,total_pos,ground_truth):
    """
    Calculate the mAP for classification.
    """
    # TODO (Q2.3): Calculate mAP on test set.
    # Feel free to write necessary function parameters.
    # print("calc_map() Input shapes = ", true_pos.shape,total_pos.shape,ground_truth.shape)
    
     #TP/ TP +FN (Total positives)
    # Take into account zero division errors
    recall = np.divide(true_pos,ground_truth, where = ground_truth!=0, out = np.zeros_like(true_pos))
    
    #TP / TP + FP
    precision = np.divide(true_pos,total_pos, where = total_pos != 0, out = np.zeros_like(true_pos))

    #Area under curve
    average_precision = precision*recall
    # print("Shape of AP array = ", average_precision.shape)
    return average_precision

### Q2.3
### The output we obtain from the model is shaped (n_roi,n_classes). Thus we get class scores for each region proposal for each class 
### For each class we individually do the following - 
### 1. Check if atleast one score is above the given threshold
### 2. Perfrom non maximum suppression to obtain our predicted positive bounding boxes
### 3. use IOU calculation for each predicted bounding box with the ground truth bounding 
###    boxes to get true positives and false positives.  
def test_model(model, val_loader=None, thresh=0.05,epoch = None):
    """
    Tests the networks and visualizes the detections
    :param thresh: Confidence threshold
    """
    with torch.no_grad():
        print("#####################  IN TESTING () #########################")
        losses = AverageMeter()
        # Initialze  arrays to zero for each class
        ground_truth = np.zeros(20)
        TP = np.zeros(20) # true positves
        total_pos = np.zeros(20) # total positives
        for iter, data in enumerate(val_loader):

            # print("test Iter = ", iter)
            image = data['image'].to(device)
            target = data['label'].to(device)
            wgt = data['wgt']
            rois = data['rois'].to(device)* 512 # Resize to pixels
            gt_boxes = data['gt_boxes']
            gt_class_list = data['gt_classes']
            # print("Gt boxes, type and len = ", type(gt_boxes), len(gt_boxes))
            # print("Gt class list, type and len = ", type(gt_class_list), len(gt_class_list))

            # Initialize the ground truth for each class present with a corresponding bounding box.
            for j in range(len(gt_boxes)):
                ground_truth[gt_class_list[j]] += 1

            # TODO (Q2.3): perform forward pass, compute cls_probs
            cls_prob = model(image, rois, target)
            loss = model.loss 
            losses.update(loss.item())
            predicted_class_list = []
            predicted_bbox_list = []
            # TODO (Q2.3): Iterate over each class (follow comments)
            for c in range(len(CLASS_NAMES)):
                # get valid rois and cls_scores based on thresh
                cls = cls_prob[:,c]
                squeeze_roi = torch.squeeze(rois).int() # Convert to int to prevent runtime error: Both inputs should be same type.
                if (torch.max(cls) >= thresh):
                    # use NMS to get boxes and scores
                    bounding_boxes, scores = nms(squeeze_roi, cls, thresh)
                    total_pos[c] += len(scores) 
                    # List for storing classses of ppredicted bboxes. Used for wandb logging.
                    class_list = []
                    for box in bounding_boxes:
                        for i, gt in enumerate(gt_boxes):
                            # Only first detection should be considered.
                            if gt_class_list[i].item() == c and i not in class_list: 
                                # Normalize box to get both boxes in the same format
                                normalized_box = box / 512
                                if iou(normalized_box,gt)>0.3: # Threshold 0.3 given.
                                    # Add box if iou is > thrshold with given gt.
                                    predicted_bbox_list.append(normalized_box)
                                    predicted_class_list.append(c)
                                    # Increase true positiv count 
                                    TP[c]+=1
                                    class_list.append(i)
                                    break

            # TODO (Q2.3): visualize bounding box predictions when required
            # Visualize only for first and last epoch
            if (epoch == 0 or epoch ==5):
                # Visualizing 20 images (select 10)
                if iter < 20:
                    img = wandb.Image(image, boxes= {
                        "predictions": {
                            "box_data":get_box_data(predicted_class_list,predicted_bbox_list),
                            "class_labels": class_id_to_label
                        },
                    })
                    wandb.log({"bbox_prediction/epoch " + str(epoch) : img})
        AP = calculate_map(TP, total_pos,ground_truth)
        mAP = np.mean(AP)
        wandb.log({"validation/loss": losses.avg})
        wandb.log({"validation/mAP": mAP})
        for k in range(len(CLASS_NAMES)):
            wandb.log({"validation/AP - Class " + str(CLASS_NAMES[k]): AP[k]})
        return AP


def train_model(model, train_loader=None, val_loader=None, optimizer=None, args=None):
    """
    Trains the network, runs evaluation and visualizes the detections
    """
    print("##############  IN TRAIN()  ###########")
    # Initialize training variables
    train_loss = 0
    step_cnt = 0
    losses = AverageMeter()
    #print(args)
    for epoch in range(args.epochs):
        for iter, data in enumerate(train_loader):
            # Collate function not needed as batch size is only equal to one
            # print("##############  TRAIN LOADER ITER =   ###########", iter)
            # print("len data =- ", len(data))
            # TODO (Q2.2): get one batch and perform forward pass
            # one batch = data for one image
            image = data['image'].to(device)
            target = data['label'].to(device)
            wgt = data['wgt']
            rois = data['rois'].to(device) * 512
            # print("Image shape = ", image.shape)
            # print("ROIS shape = ", rois.shape)

            # TODO (Q2.2): perform forward pass
            # take care that proposal values should be in pixels
            # Convert inputs to cuda if training on GPU
            model.train()
            output = model(image, rois, target)

            # backward pass and update
            loss = model.loss
            losses.update(loss.item())
            train_loss += loss.item()
            step_cnt += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # TODO (Q2.2): evaluate the model every N iterations (N defined in handout)
            # Add wandb logging wherever necessary
            if step_cnt % args.val_interval == 0: 
                wandb.log({'train/loss': losses.avg})   
                # print("Args interval = ", args.val_interval)
             
        # TODO (Q2.4): Perform all visualizations here
        # The intervals for different things are defined in the handout
        model.eval()
        # Send epoch to test model to print during logging.
        ap = test_model(model, val_loader, thresh=0.05, epoch = epoch)
        model.train()
        # print("mAP valuie = ", np.mean(ap))

def main():
    """
    Creates dataloaders, network, and calls the trainer
    """
    args = parser.parse_args()
    # TODO (Q2.2): Load datasets and create dataloaders
    # Initialize wandb logger
    if args.use_wandb:
        wandb.init(project="COLAB-VLR-Task2", reinit=True)

    train_dataset = VOCDataset('trainval', top_n=300,image_size = 512)
    val_dataset = VOCDataset('test', top_n=300, image_size = 512)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,   # batchsize is one for this implementation
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=None,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True)

    # Create network and initialize
    net = WSDDN(classes=CLASS_NAMES)
    print(net)

    if os.path.exists('pretrained_alexnet.pkl'):
        pret_net = pkl.load(open('pretrained_alexnet.pkl', 'rb'))
    else:
        pret_net = model_zoo.load_url(
            'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
        pkl.dump(pret_net,
        open('pretrained_alexnet.pkl', 'wb'), pkl.HIGHEST_PROTOCOL)
    own_state = net.state_dict()

    for name, param in pret_net.items():
        print(name)
        if name not in own_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        try:
            own_state[name].copy_(param)
            print('Copied {}'.format(name))
        except:
            print('Did not find {}'.format(name))
            continue
    print("##########################  FINISHED INITIALIZING #######################################")
    # Move model to GPU and set train mode
    net.load_state_dict(own_state)
    net.to(device)
    net.train()

    # TODO (Q2.2): Freeze AlexNet layers since we are loading a pretrained model
    for p in net.features:
        p.requires_grad_ = False

    # TODO (Q2.2): Create optimizer only for network parameters that are trainable
    optimizer = torch.optim.SGD(net.parameters(), lr = 0.0001, momentum=0.9, weight_decay= 1e-4)
    print("##########################  BEFORE TRAIN  #######################################")
    # Training
    train_model(net, train_loader, val_loader, optimizer, args)

if __name__ == '__main__':
    print("Device name is = ", device)
    main()
