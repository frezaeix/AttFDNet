# from .voc import VOCDetection, AnnotationTransform, detection_collate, VOC_CLASSES
from .voc0712 import VOCDetection, VOCDetection_fewshot, VOCDetection_fewshot_new, AnnotationTransform, detection_collate, VOC_CLASSES
from .coco import COCODetection
from .data_augment import *
from .config import *
