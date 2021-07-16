import torch
import torchvision
import cv2
import os
import faster_rcnn.visualize.draw as draw
from torchvision.transforms import functional as F


class ImageVisualizer(object):
    def __init__(self, output_path, json_path, model_weight, threshold):
        model = 'fasterrcnn_resnet50_fpn'
        num_classes = 91
        self.device = 'cuda'
        self.threshold = threshold
        pretrained = False

        model = torchvision.models.detection.__dict__[model](num_classes=num_classes, pretrained=pretrained)
        checkpoint = torch.load(model_weight)
        model.load_state_dict(checkpoint['model'])
        model.to(self.device)
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
        ori_image = cv2.imread(image_path)
        image = F.to_tensor(ori_image)
        image = image.unsqueeze(0)
        image = image.to(self.device)

        detections = self.model(image)

        boxes = detections[0]['boxes'].tolist()
        labels = detections[0]['labels'].tolist()
        scores = detections[0]['scores'].tolist()

        bbox_image = draw.draw_bboxes(ori_image, boxes, labels, scores, self.category, self.threshold)
        cv2.imwrite(os.path.join(self.output_path, image_path.split('\\')[-1]), bbox_image)


