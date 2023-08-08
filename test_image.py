
import os
import yaml
import argparse
import numpy as np
import glob
from onnxruntime import InferenceSession

from preprocess import Compose
import cv2

def draw_bbox(img,bbox_list,classes):
    for x,bbox in enumerate(bbox_list):
        start_point = (int(bbox[0]),int(bbox[1]))
        end_point = (int(bbox[2]),int(bbox[3]))
        cv2.rectangle(img, start_point, end_point, (255,0,0), 2)
        # if not face:
            # cv2.putText(img, classes[x], (start_point[0] - 2, start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    
    return img

# Global dictionary
SUPPORT_MODELS = {
    'YOLO', 'PPYOLOE', 'RCNN', 'SSD', 'Face', 'FCOS', 'SOLOv2', 'TTFNet',
    'S2ANet', 'JDE', 'FairMOT', 'DeepSORT', 'GFL', 'PicoDet', 'CenterNet',
    'TOOD', 'RetinaNet', 'StrongBaseline', 'STGCN', 'YOLOX', 'HRNet'
}

class PredictConfig(object):
    """set config of preprocess, postprocess and visualize
    Args:
        infer_config (str): path of infer_cfg.yml
    """

    def __init__(self, infer_config):
        # parsing Yaml config for Preprocess
        with open(infer_config) as f:
            yml_conf = yaml.safe_load(f)
        self.check_model(yml_conf)
        self.arch = yml_conf['arch']
        self.preprocess_infos = yml_conf['Preprocess']
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.label_list = yml_conf['label_list']
        self.use_dynamic_shape = yml_conf['use_dynamic_shape']
        self.draw_threshold = yml_conf.get("draw_threshold", 0.5)
        self.mask = yml_conf.get("mask", False)
        self.tracker = yml_conf.get("tracker", None)
        self.nms = yml_conf.get("NMS", None)
        self.fpn_stride = yml_conf.get("fpn_stride", None)
        if self.arch == 'RCNN' and yml_conf.get('export_onnx', False):
            print(
                'The RCNN export model is used for ONNX and it only supports batch_size = 1'
            )
        self.print_config()

    def check_model(self, yml_conf):
        """
        Raises:
            ValueError: loaded model not in supported model type
        """
        for support_model in SUPPORT_MODELS:
            if support_model in yml_conf['arch']:
                return True
        raise ValueError("Unsupported arch: {}, expect {}".format(yml_conf[
            'arch'], SUPPORT_MODELS))

    def print_config(self):
        print('-----------  Model Configuration -----------')
        print('%s: %s' % ('Model Arch', self.arch))
        print('%s: ' % ('Transform Order'))
        for op_info in self.preprocess_infos:
            print('--%s: %s' % ('transform op', op_info['type']))
        print('--------------------------------------------')

def predict_image(infer_config, predictor, img_list):
    # load preprocess transforms
    bbox_list = []
    class_list = []
    transforms = Compose(infer_config.preprocess_infos)
    # predict image
    # for img_path in img_list:
   
    inputs = transforms(img_list)
    

    inputs_name = [var.name for var in predictor.get_inputs()]
    inputs = {k: inputs[k][None, ] for k in inputs_name}
    
    outputs = predictor.run(output_names=None, input_feed=inputs)

    print("ONNXRuntime predict: ")
    if infer_config.arch in ["HRNet"]:
        print(np.array(outputs[0]))
    else:
        bboxes = outputs[0]
        for bbox in bboxes:
            if bbox[0] > -1 and bbox[1] > infer_config.draw_threshold:
                bbox_list.append(bbox[2:])
                class_list.append(bbox[0])
                print(f"{int(bbox[0])} {bbox[1]} "
                        f"{bbox[2]} {bbox[3]} {bbox[4]} {bbox[5]}")
    return bbox_list,class_list


cuda = True
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
predictor = InferenceSession("pphuman.onnx", providers=providers)


image = cv2.imread("human.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# load infer config
infer_config = PredictConfig("infer_cfg.yml")

bbox,classes = predict_image(infer_config, predictor, image)

image = draw_bbox(image,bbox,classes)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imwrite("rframe.png",image)
#cv2.waitKey(0)

        
