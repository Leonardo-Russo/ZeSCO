import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch
import torch.nn as nn


class IndoorOutdoorClassifier(nn.Module):
    def __init__(self):
        super(IndoorOutdoorClassifier, self).__init__()
        model_name = "prithivMLmods/IndoorOutdoorNet"  # Updated model name
        self.model = SiglipForImageClassification.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def forward(self, image, debug=False):
        """Predicts whether an image is Indoor or Outdoor."""
        if isinstance(image, torch.Tensor):
            inputs = {"pixel_values": image}
        else:
            inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

        labels = {
            "0": "Indoor", "1": "Outdoor"
        }
        predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}

        return predictions