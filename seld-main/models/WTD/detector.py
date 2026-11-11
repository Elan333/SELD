import numpy as np
import torch
import torchvision.transforms as transforms

from .dinov3_attention import MultiLayerHook, fuse_attention
from .lcm import compute_lcm, preprocess_sobel
from .morphology import detect_with_strict_morphology, saliency_contrast_ratio
from .nms import global_nms
from .roi import crop_roi, find_top_k_rois


class WeakTargetDetector:
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.device = device
        self.lcm_scales = config["lcm"]["scales"]
        self.peak_sigma = config["lcm"]["peak_sigma"]
        self.roi_size = config["roi"]["size"]
        self.top_k = config["roi"]["top_k"]
        self.nms_iou = config["detection"]["nms_iou"]
        self.morphology = dict(config["morphology"], roi_size=self.roi_size)
        self.hook = MultiLayerHook(
            model.blocks,
            config["attention"]["layers"],
            {int(head): weight for head, weight in config["attention"]["head_weights"].items()},
            config["attention"]["heads"],
            config["attention"]["storage_tokens"],
        )
        self.transform = transforms.Compose([
            transforms.Resize((self.roi_size, self.roi_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @property
    def layers(self):
        return self.hook.layers

    def detect(self, image):
        processed = preprocess_sobel(image)
        gray = np.array(processed.convert("L"))
        lcm = compute_lcm(gray, self.lcm_scales)
        rois = find_top_k_rois(lcm, self.top_k, self.roi_size, self.peak_sigma)
        detections = []
        if rois:
            batch = torch.stack([
                self.transform(crop_roi(processed, roi, self.roi_size)) for roi in rois
            ]).to(self.device)
            self.hook.clear()
            with torch.no_grad():
                self.model(batch)
            saliency = fuse_attention(self.hook.attention_maps(), *self.layers)
            for index, roi in enumerate(rois):
                detections.extend(detect_with_strict_morphology(
                    saliency[index],
                    saliency_contrast_ratio(saliency[index]),
                    roi["offset"],
                    self.morphology,
                ))
        return global_nms(detections, self.nms_iou), rois

    def close(self):
        self.hook.remove()
