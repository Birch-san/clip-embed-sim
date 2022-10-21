import json
from glob import iglob
from os import path
from pathlib import Path
from typing import Callable, List, NamedTuple, TypeVar, Union

import open_clip
import torch
from open_clip import CLIP as OpenCLIP
from PIL import Image
from torch import Tensor, no_grad
from torch.nn import functional as F
from typing_extensions import TypeAlias

device=torch.device('mps')

T = TypeVar('T')
Decorator: TypeAlias = Callable[[T], T]
TensorDecorator: TypeAlias = Decorator[Tensor]

EncodeText: TypeAlias = TensorDecorator
ImgPreprocess: TypeAlias = Callable[[Image.Image], Tensor]
DeviceDescriptor: TypeAlias = Union[str, torch.device]

class Topk(NamedTuple):
  values: Tensor
  indices: Tensor

def with_mps_mitigation(encode_text: EncodeText) -> EncodeText:
  def mitigated_encode_text(captions: Tensor) -> Tensor:
    if captions.device.type == 'mps' and captions.size(dim=0) > 1:
      # run serially (instead of batching), then concat afterward.
      # to avoid MPS bug on pytorch 1.12.1 where layernorm breaks if captions.size(dim=0) > 1
      return torch.cat([
        encode_text(caption) for caption in captions.split(1, dim=0)
      ])
    return encode_text(captions)
  return mitigated_encode_text

class WrappedModel:
  model_name: str
  model_ver: str
  model: OpenCLIP
  device: DeviceDescriptor
  img_preprocess: ImgPreprocess
  def __init__(
    self,
    model_name: str,
    model_ver: str,
    device: DeviceDescriptor = 'cpu'
  ) -> None:
    self.model_name = model_name
    self.model_ver = model_ver
    self.device = device
    model, _, val_preprocess = open_clip.create_model_and_transforms(model_name, model_ver, device=device)
    model.requires_grad_(False)
    self.model = model
    self.img_preprocess = val_preprocess

class TestSubject:
  wrapped_model: WrappedModel
  coarse_class_text_features: Tensor
  def __init__(
    self,
    wrapped_model: WrappedModel,
    coarse_class_tokens: Tensor
  ) -> None:
    self.wrapped_model=wrapped_model
    with no_grad():
      encode_text: EncodeText = with_mps_mitigation(wrapped_model.model.encode_text)
      self.coarse_class_text_features: Tensor = encode_text(coarse_class_tokens)
  
  def get_topk(
    self,
    pil_img: Image.Image,
    caption_tokens: Tensor,
    k = 5
  ) -> Topk:
    model: OpenCLIP = self.wrapped_model.model
    encode_text: EncodeText = with_mps_mitigation(model.encode_text)
    device: DeviceDescriptor = self.wrapped_model.device
    img: Tensor = self.wrapped_model.img_preprocess(pil_img) # [3, 244, 244]
    img_batch: Tensor = img.to(device).unsqueeze(0) # [1, 3, 244, 244]

    # a shorthand is available, but we avoid due to MPS bug on pytorch 1.12.1 where layernorm breaks if encode_text() called with tokens.size(dim=0) > 1
    # image_features, text_features, logit_scale_exp = laion_model.forward(laion_img_batch, laion_tokens.to(device)) # ([1, 512], [1, 512], [])

    with no_grad():
      caption_text_features: Tensor = encode_text(caption_tokens) # [1, 512]
      image_features: Tensor = model.encode_image(img_batch).to(device) # [n, 512]

    text_features = torch.cat([caption_text_features, self.coarse_class_text_features])
    text_features = F.normalize(text_features, dim=-1)
    image_features = F.normalize(image_features, dim=-1)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (model.logit_scale.exp() * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(k)
    return Topk(values=values, indices=indices)

class SubjectTopk(NamedTuple):
  subject: TestSubject
  topk: Topk

models: List[WrappedModel] = [
  WrappedModel(
    device=device,
    model_name='ViT-H-14',
    model_ver='laion2b_s32b_b79k'
  ), # big
  WrappedModel(
    device=device,
    model_name='ViT-B-32',
    model_ver='laion2b_s34b_b79k'
  ),
  WrappedModel(
    device=device,
    model_name='ViT-L-14',
    model_ver='openai'
  ),
]

img_to_caption = json.load(open('img_to_caption.json', 'r'))

coarse_classes: List[str] = ['a painting', 'trending on artstation']
coarse_class_tokens: Tensor = open_clip.tokenize(coarse_classes).to(device) # [2, 77]

subjects: List[TestSubject] = [TestSubject(
  wrapped_model=wrapped_model,
  coarse_class_tokens=coarse_class_tokens,
) for wrapped_model in models]

for filename in iglob('square/*.jpg'):
  assert path.isfile(filename)
  pil_img: Image.Image = Image.open(filename)
  leafname: str = Path(filename).name
  assert leafname in img_to_caption
  caption: str = img_to_caption[leafname]
  classes: List[str] = [caption, *coarse_classes]
  caption_tokens: Tensor = open_clip.tokenize(caption).to(device) # [1, 77]

  print(f"Assessing CLIP image-text similarity for image captioned '{caption}'...")
  subject_topks: List[SubjectTopk] = [
    SubjectTopk(
      subject=subject,
      topk=subject.get_topk(
        pil_img=pil_img,
        caption_tokens=caption_tokens
      )
    ) for subject in subjects
  ]

  for subject, topk in subject_topks:
    print(f"  Top predictions for {subject.wrapped_model.model_name} {subject.wrapped_model.model_ver}:")
    values, indices = topk
    for value, index in zip(values, indices):
      print(f"    {classes[index]:>30s}: {100 * value.item():.2f}%")
  print('')