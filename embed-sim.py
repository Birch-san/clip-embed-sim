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

# attempt to load OpenAI model using openclip
# laion_model_name = 'ViT-L-14'
# laion_model_ver = 'openai'

# laion_model: OpenCLIP = open_clip.create_model(laion_model_name, laion_model_ver, device=device)
laion_model, _, laion_val_preprocess = open_clip.create_model_and_transforms(laion_model_name, laion_model_ver, device=device)
laion_model.requires_grad_(False)
clip_normalize = transforms.Normalize(mean=laion_model.visual.image_mean, std=laion_model.visual.image_std)

img_to_caption = json.load(open('img_to_caption.json', 'r'))

coarse_classes = ['a painting', 'trending on artstation']
laion_coarse_class_tokens: Tensor = open_clip.tokenize(coarse_classes) # [2, 77]
laion_coarse_class_text_features = laion_model.cpu().encode_text(laion_coarse_class_tokens).to(device) # move model to CPU to avoid MPS bug where layernorm breaks if tokens.size(dim=0) > 1
laion_model.to(device)
logit_scale_exp = laion_model.logit_scale.exp()

for filename in iglob('square/*.jpg'):
  assert path.isfile(filename)
  pil_img: Image.Image = Image.open(filename)

  laion_img: Tensor = laion_val_preprocess(pil_img) # [3, 244, 244]
  laion_img_batch: Tensor = laion_img.to(device).unsqueeze(0) # [1, 3, 244, 244]
  
  leafname: str = Path(filename).name
  assert leafname in img_to_caption
  caption: str = img_to_caption[leafname]
  laion_caption_tokens: Tensor = open_clip.tokenize(caption).to(device) # [1, 77]
  laion_caption_text_features: Tensor = laion_model.encode_text(laion_caption_tokens) # [1, 512]
  # laion_tokens: Tensor = torch.cat([laion_caption_tokens, laion_coarse_class_tokens])
  # laion_coarse_class_text_features
  text_features = torch.cat([laion_caption_text_features, laion_coarse_class_text_features])
  text_features = F.normalize(text_features, dim=-1)
  image_features: Tensor = laion_model.encode_image(laion_img_batch).to(device) # [n, 512]
  image_features = F.normalize(image_features, dim=-1)

  # image_features, text_features, logit_scale_exp = laion_model.forward(laion_img_batch, laion_tokens.to(device)) # ([1, 512], [1, 512], [])

  image_features /= image_features.norm(dim=-1, keepdim=True)
  text_features /= text_features.norm(dim=-1, keepdim=True)
  similarity = (logit_scale_exp * image_features @ text_features.T).softmax(dim=-1)
  values, indices = similarity[0].topk(3)

  # Print the result
  print("\nTop predictions:\n")
  classes = [caption, *coarse_classes]
  for value, index in zip(values, indices):
    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
  print("Done")


  # laion_target_embed: Tensor = F.normalize(laion_encoded)

# model, preprocess = clip.load("ViT-B/32", device=device, jit=jit)