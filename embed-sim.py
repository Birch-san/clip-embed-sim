import json
from glob import iglob
from os import path
from pathlib import Path

import clip
import open_clip
import torch
from open_clip import CLIP as OpenCLIP
from torch import Tensor, no_grad
from transformers import CLIPTextModel, CLIPTokenizer

device=torch.device('mps')
jit=False

openai_clip_ver = 'openai/clip-vit-large-patch14'
tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(openai_clip_ver)
transformer: CLIPTextModel = CLIPTextModel.from_pretrained(openai_clip_ver)
transformer.requires_grad_(False)

# big:
# openclip_model_name = 'ViT-H-14'
# openclip_model_ver = 'laion2b_s32b_b79k'

# less big:
openclip_model_name = 'ViT-B-32'
openclip_model_ver = 'laion2b_s34b_b79k'

clip_model: OpenCLIP = open_clip.create_model(openclip_model_name, openclip_model_ver, device=device)
clip_model.requires_grad_(False)

img_to_caption = json.load(open('img_to_caption.json', 'r'))

with no_grad():
  for filename in iglob('square/*.jpg'):
    assert path.isfile(filename)
    leafname: str = Path(filename).name
    assert leafname in img_to_caption
    caption: str = img_to_caption[leafname]
    tokens: Tensor = open_clip.tokenize(caption).to(device)
    encoded: Tensor = clip_model.encode_text(tokens)

# model, preprocess = clip.load("ViT-B/32", device=device, jit=jit)