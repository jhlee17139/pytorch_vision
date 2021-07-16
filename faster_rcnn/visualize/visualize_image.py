import torch
import torchvision
import cv2
import faster_rcnn.visualize.draw as draw


class ImageVisualizer(object):
    def __init__(self, output_path, json_path):
        model = 'fasterrcnn_resnet50_fpn'
        num_classes = 91
        device = 'cuda'
        pretrained = False
        model_weight = ''

        model = torchvision.models.detection.__dict__[model](num_classes=num_classes, pretrained=pretrained)
        checkpoint = torch.load(model_weight)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        model.eval()

        self.model = model
        self.output_path = output_path
        self.category = draw.get_coco_labels(json_path)

    def detect_image(self, image_path):
        '''
        :param image_path:
        :return: detections = {
            "boxes": ,
            "labels": ,
            "scores": ,
        }
        '''
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image)
        _, detections = self.model(image)






