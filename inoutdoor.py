import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch


class IndoorOutdoorClassifier:
    def __init__(self):
        # Load model and processor
        model_name = "prithivMLmods/IndoorOutdoorNet"  # Updated model name
        self.model = SiglipForImageClassification.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def forward(self, image, debug=False):
        """Predicts whether an image is Indoor or Outdoor."""
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


# Create Gradio interface
iface = gr.Interface(
    fn=IndoorOutdoorClassifier(),
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="IndoorOutdoorNet",
    description="Upload an image to classify it as Indoor or Outdoor."
)

if __name__ == "__main__":
    iface.launch()
