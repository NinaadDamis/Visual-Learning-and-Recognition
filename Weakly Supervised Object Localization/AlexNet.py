import torch.nn as nn
import torchvision.models as models
import torch

def init_weights(module):
        #https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch
        if isinstance(module, nn.Conv2d):
            # nn.init.xavier_uniform_(module.weight.data,gain =1)
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                # Zero initialize bias
                nn.init.constant_(module.bias.data, 0)

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class LocalizerAlexNet(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNet, self).__init__()

        alexnet = models.alexnet()

        # weight = models.AlexNet_Weights
        # alexnet = models.alexnet(weights = weight.DEFAULT)
        # alexnet_dict = list(alexnet.state_dict().items())
        # AlexNet State Dict
        
        # features.0.weight
        # features.0.bias
        # features.3.weight
        # features.3.bias
        # features.6.weight
        # features.6.bias
        # features.8.weight
        # features.8.bias
        # features.10.weight
        # features.10.bias
        # classifier.1.weight
        # classifier.1.bias
        # classifier.4.weight
        # classifier.4.bias
        # classifier.6.weight
        # classifier.6.bias

        pretrained_params =torch.hub.load_state_dict_from_url(model_urls['alexnet'])
        alexnet.load_state_dict(pretrained_params)

        # TODO (Q1.1): Define model
        self.features=nn.Sequential(*list(alexnet.children()))[0][:12]

        # Freeze weights
        # self.features.requires_grad_(False)
        for p in self.features:
            p.requires_grad = False

        self.classifier = nn.Sequential(

            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1,stride = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1,stride=1),
        )

        self.classifier.apply(init_weights)
        # self.global_pool = nn.AdaptiveMaxPool2d(kernel_size=20)

    def forward(self, x):
        # TODO (Q1.1): Define forward pass
        features = self.features(x)
        output = self.classifier(features)
        # x = self.global_pool(x)
        return output


class LocalizerAlexNetRobust(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNetRobust, self).__init__()
        # TODO (Q1.7): Define model
        alexnet = models.alexnet()
        pretrained_params=torch.hub.load_state_dict_from_url(model_urls['alexnet'])

        # Load weights
        alexnet.load_state_dict(pretrained_params)
        self.features=nn.Sequential(*list(alexnet.children()))[0][:12]

        # Freeze weights for features layers.
        for p in self.features:
            p.requires_grad = False

        self.classifier = nn.Sequential(

            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1,stride = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1,stride=1),
        )

        self.classifier.apply(init_weights)

        # self.global_pool = nn.AdaptiveAvvgPool2d((1,1))


    def forward(self, x):
        # TODO (Q1.7): Define forward pass
        features = self.features(x)
        output   = self.classifier(features)
        # x = self.global_pool(output)
        return output


def localizer_alexnet(pretrained=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        print("Initializing loacalizer_alexnet ()")
        model = LocalizerAlexNet(**kwargs)
    else:
        print("Error : Pretrained = False")
    # TODO (Q1.3): Initialize weights based on whether it is pretrained or not

    return model


def localizer_alexnet_robust(pretrained=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        print("Initializing robust localizer_alexnet ()")
        model = LocalizerAlexNetRobust(**kwargs)
    else:
        print("Error : Pretrained = False")
    # TODO (Q1.7): Initialize weights based on whether it is pretrained or not

    return model
