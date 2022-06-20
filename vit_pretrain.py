import transformers
from transformers import ViTModel, ViTConfig
from transformers import ViTFeatureExtractor, ViTModel
import torch
from torchvision import transforms
from PIL import Image, ImageOps
import os
from os.path import join


image = Image.open(join('images', 'IMG_9968.jpg'))
image = ImageOps.exif_transpose(image)
w, h = image.size
print("PIL Image width: {}, height: {}".format(w, h))
trans = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])
sample = trans(image)
sample = sample.unsqueeze(0)


feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

inputs = feature_extractor(sample, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
print(list(last_hidden_states.shape))

# Accessing the model configuration
configuration = model.config
print(configuration)
