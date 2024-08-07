import numpy as np
import torch
import torchvision
import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torchvision.ops import roi_pool, roi_align

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class WSDDN(nn.Module):
    n_classes = 20
    classes = np.asarray([
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ])

    def __init__(self, classes=None):
        super(WSDDN, self).__init__()

        if classes is not None:
            self.classes = classes
            self.n_classes = len(classes)
            print(classes)

        # TODO (Q2.1): Define the WSDDN model
        # print(*list(AlexNet.children()))
        # Same as alexnet
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 192, kernel_size=5, stride = 1,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2,dilation=1,ceil_mode=False),
            nn.Conv2d(192, 384, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3,stride = 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3,stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # self.features = nn.Sequential(*list(AlexNet.children()))[0][:12]
        # self.roi_pool = roi_pool(output_size=(6,6), spatial_scale=31.0/512) # Output of features is sized 31,31.
        # Dropout = 0.4
        # Classifier is similar to alexnet classifier - redefined to make output = 4096.
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(9216, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(4096, 4096),
            nn.ReLU(),
        )

        self.score_fc = nn.Linear(4096, 20) # Output = class size
        self.bbox_fc = nn.Linear(4096, 20)

        # loss
        self.loss_function = torch.nn.BCELoss(reduction='sum')
        self.cross_entropy = None
        self.training = True

    @property
    def loss(self):
        return self.cross_entropy

    def forward(self,
                image,
                rois=None,
                gt_vec=None,
                ):


        # TODO (Q2.1): Use image and rois as input
        # compute cls_prob which are N_roi X 20 scores
        features = self.features(image)
        # print("Shape features = ", features.shape) # Use to get spatial scale for roi pool
        # print("Shape ROIS = ", rois.shape)
        rois = [torch.squeeze(rois).float().to("cuda")]
        # print("Shape ROIS after squeeze = ", rois.shape) Makes it 2D
        roi_output = roi_pool(features,rois,output_size=(6,6), spatial_scale= 31.0/512.0)
        # Flatten for fc layer
        roi_output = roi_output.view(-1,9216)

        classifier_out = self.classifier(roi_output)
        score_out = self.score_fc(classifier_out)
        bbox_out  = self.bbox_fc(classifier_out)
        score_softmax = F.softmax(score_out, dim = 1)
        bbox_softmax  = F.softmax(bbox_out, dim = 0)
        # print("Softmax shapes = ", score_softmax.shape)
        cls_prob = score_softmax*bbox_softmax
        # print("Shape cls prob = ", cls_prob.shape)
        if self.training:
            # print("########################  BUILT LOSS ######################################")
            label_vec = gt_vec.view(self.n_classes, -1)
            self.cross_entropy = self.build_loss(cls_prob, label_vec)
        return cls_prob

    def build_loss(self, cls_prob, label_vec):
        """Computes the loss

        :cls_prob: N_roix20 output scores
        :label_vec: 1x20 one hot label vector
        :returns: loss

        """
        # TODO (Q2.1): Compute the appropriate loss using the cls_prob
        # that is the output of forward()
        # Checkout forward() to see how it is called
        cls_vec = torch.sum(cls_prob, dim=0)
        # print("Shape cls vec = ", cls_vec.shape)
        cls_vec = cls_vec.view(20,1) # Change from 20 to 20,1 to convert shape to that of label
        cls_vec = torch.clamp(cls_vec,0,1) # Clamp bw 0 to 1
        loss = self.loss_function(cls_vec, label_vec)
        return loss
