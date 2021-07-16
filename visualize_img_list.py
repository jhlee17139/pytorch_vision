import os
from faster_rcnn.visualize.visualize_image import ImageVisualizer


if __name__ == '__main__':
    output_path = '.\\output\\visualize\\image'
    input_path = '.\\input\\image'
    json_path = 'G:\\vsgg_ws\\coco_dataset\\annotations\\instances_val2017.json'
    model_weight = '.\\output\\weight\\model_0.pth'
    threshold = 0.75

    print("1. load detector")
    image_visualize = ImageVisualizer(output_path, json_path, model_weight, threshold)

    print("2. coco label list")
    print(image_visualize.category)

    image_path_list = os.listdir(input_path)
    print("3. input image len : {}".format(len(image_path_list)))

    tmp = []
    for image_path in image_path_list:
        tmp.append(os.path.join(input_path, image_path))
    image_path_list = tmp

    for image_path in image_path_list:
        image_visualize.detect_image(image_path)




