import torch
import torchvision
from PIL import Image


class ImageVisualizer(object):
    def __init__(self):
        model = 'fasterrcnn_resnet50_fpn'
        num_classes = 80
        device = 'cuda'
        pretrained = False
        model_weight = ''

        model = torchvision.models.detection.__dict__[model](num_classes=num_classes, pretrained=pretrained)
        checkpoint = torch.load(model_weight)
        model.load_state_dict(checkpoint['model'])
        model.to(device)

        self.model = model

    def 
