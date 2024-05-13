from flask import Flask, request
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import requests
import torch.nn as nn
import numpy as np

app = Flask(__name__)

@app.route('/api/get_url', methods=['POST'])
def get_url():
    data = request.get_json()
    url = data['url']

    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0]
    pred_seg_np = pred_seg.numpy()

    unique_labels = np.unique(pred_seg_np)

    for label in unique_labels:
        mask = np.where(pred_seg_np == label, 1, 0).astype(np.uint8)
        segmented_part = np.array(image) * mask[:, :, None]
        segmented_image = Image.fromarray(segmented_part)
        segmented_image.save(f"segmented_{label}.png")

    return "Segmentation completed."

if __name__ == "__main__":
    app.run(debug=True)