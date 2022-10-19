import json
from glob import iglob
from os import path
from pathlib import Path

import clip
import open_clip
import torch
from open_clip import CLIP as OpenCLIP
from PIL import Image
from torch import Tensor, no_grad
from torch.nn import functional as F
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel

device=torch.device('mps')
jit=False

oai_clip_ver = 'openai/clip-vit-large-patch14'
oai_tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(oai_clip_ver)
oai_text: CLIPTextModel = CLIPTextModel.from_pretrained(oai_clip_ver)
oai_text.requires_grad_(False)
oai_vision: CLIPVisionModel = CLIPVisionModel.from_pretrained(oai_clip_ver)
oai_vision.requires_grad_(False)

# oai_clip_model, oai_clip_transform = clip.load(name='ViT-L/14', device=device, jit=jit)

# big:
# laion_model_name = 'ViT-H-14'
# laion_model_ver = 'laion2b_s32b_b79k'

# less big:
laion_model_name = 'ViT-B-32'
laion_model_ver = 'laion2b_s34b_b79k'

# laion_model: OpenCLIP = open_clip.create_model(laion_model_name, laion_model_ver, device=device)
laion_model, _, laion_val_preprocess = open_clip.create_model_and_transforms(laion_model_name, laion_model_ver, device=device)
laion_model.requires_grad_(False)
clip_normalize = transforms.Normalize(mean=laion_model.visual.image_mean, std=laion_model.visual.image_std)

img_to_caption = json.load(open('img_to_caption.json', 'r'))

with no_grad():
  for filename in iglob('square/*.jpg'):
    assert path.isfile(filename)
    pil_img: Image.Image = Image.open("CLIP.png")

    laion_img: Tensor = laion_val_preprocess(pil_img)
    laion_img_batch: Tensor = laion_img.unsqueeze(0)
    
    leafname: str = Path(filename).name
    assert leafname in img_to_caption
    caption: str = img_to_caption[leafname]
    laion_tokens: Tensor = open_clip.tokenize(caption).to(device)
    laion_encoded: Tensor = laion_model.encode_text(laion_tokens)
    laion_target_embed: Tensor = F.normalize(laion_encoded)

# model, preprocess = clip.load("ViT-B/32", device=device, jit=jit)