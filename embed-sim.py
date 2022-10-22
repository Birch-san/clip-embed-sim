import json
import re
from glob import iglob
from os import path
from pathlib import Path
from typing import Callable, Iterator, List, NamedTuple, TypeVar, Union

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

class Similarity(NamedTuple):
  raw: Topk
  softmax: Tensor

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
  
  def get_similarity(
    self,
    pil_img: Image.Image,
    caption_tokens: Tensor,
    k = 5
  ) -> Similarity:
    model: OpenCLIP = self.wrapped_model.model
    encode_text: EncodeText = with_mps_mitigation(model.encode_text)
    device: DeviceDescriptor = self.wrapped_model.device
    img: Tensor = self.wrapped_model.img_preprocess(pil_img) # [3, 244, 244]
    img_batch: Tensor = img.to(device).unsqueeze(0) # [1, 3, 244, 244]

    # a shorthand is available, but we avoid due to MPS bug on pytorch 1.12.1 where layernorm breaks if encode_text() called with tokens.size(dim=0) > 1
    # image_features, text_features, logit_scale_exp = laion_model.forward(laion_img_batch, laion_tokens.to(device)) # ([1, 512], [1, 512], [])

    with no_grad():
      caption_text_features: Tensor = encode_text(caption_tokens) # [n, 512]
      image_features: Tensor = model.encode_image(img_batch).to(device) # [m, 512]
    
    # we are never submitting more than one image at a time; eliminate batch dim to simplify slightly
    image_features = image_features.squeeze()

    text_features = torch.cat([caption_text_features, self.coarse_class_text_features])
    text_features = F.normalize(text_features, dim=-1)
    image_features = F.normalize(image_features, dim=-1)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = model.logit_scale.exp() * image_features @ text_features.T
    similarity_softmax = similarity.softmax(dim=-1)

    # transfer to CPU because MPS doesn't support topk for k>16
    similarity_topk = Topk(*similarity.cpu().topk(k))
    similarity_softmax_topk = Topk(*similarity_softmax.cpu().topk(k))

    return Similarity(raw=similarity_topk, softmax=similarity_softmax_topk)

class SubjectSimilarity(NamedTuple):
  subject: TestSubject
  similarity: Similarity

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

# the 'true' caption (e.g. alt-text which accompanied image)
img_to_caption = json.load(open('img_to_caption.json', 'r'))

coarse_classes: List[str] = ['a painting', 'trending on artstation']
coarse_class_tokens: Tensor = open_clip.tokenize(coarse_classes).to(device) # [2, 77]

subjects: List[TestSubject] = [TestSubject(
  wrapped_model=wrapped_model,
  coarse_class_tokens=coarse_class_tokens,
) for wrapped_model in models]

matcher = r"^(artist) (.*)$"
substitute_full_names: List[str] = [
]
substitute_given_names: List[str] = [
]
substitute_surnames: List[str] = [
]

def vary_caption(nominal: str) -> Iterator[str]:
  match = re.match(matcher, nominal)
  if not match:
    # unable to produce variations; caption did not match expected format
    return
  name = match.group(1)
  title = match.group(2)
  # remove name
  yield title
  # qualify
  yield f"a painting of {title}"
  # replace full name
  yield from [f"{name} {title}" for name in substitute_full_names]

  _, surname = name.split(' ')
  # change given name
  yield from [f"{given_name} {surname} {title}" for given_name in substitute_given_names]
  # change given name and surname
  for substitute_surname in substitute_surnames:
    yield from [f"{given_name} {substitute_surname} {title}" for given_name in substitute_given_names]

for filename in iglob('square/*.jpg'):
  assert path.isfile(filename)
  pil_img: Image.Image = Image.open(filename)
  leafname: str = Path(filename).name
  assert leafname in img_to_caption
  caption: str = img_to_caption[leafname]
  caption_variations: List[str] = list(vary_caption(caption))
  classes_derived_from_caption: List[str] = [caption, *caption_variations]
  
  classes: List[str] = [*classes_derived_from_caption, *coarse_classes]

  # no need to submit tokens for coarse_classes, because Subject already has them
  caption_tokens: Tensor = open_clip.tokenize(classes_derived_from_caption).to(device) # [n, 77]

  # topk=16 # maximum supported on MPS
  topk=len(classes) # nevermind we'll just do it on-CPU
  print(f"Assessing CLIP image-text similarity for image captioned,\n'{caption}'...")
  subject_similarities: List[SubjectSimilarity] = [
    SubjectSimilarity(
      subject=subject,
      similarity=subject.get_similarity(
        pil_img=pil_img,
        caption_tokens=caption_tokens,
        k=topk
      )
    ) for subject in subjects
  ]

  print(f"  Top {topk} predictions for")
  for subject, similarity in subject_similarities:
    print(f"    {subject.wrapped_model.model_name} {subject.wrapped_model.model_ver}:")
    raw, softmax = similarity
    for raw_value, raw_index, sm_value, sm_index in zip(raw.values, raw.indices, softmax.values, softmax.indices):
      print(f"      {classes[sm_index]:>60s}: {100 * sm_value.item():.2f}% {raw_value.item():.2f}")
  print('')