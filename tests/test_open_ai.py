import os
import openai
import clip
import torch

print(clip.available_models())

model, preprocess = clip.load("ViT-B/32")
model.cuda().eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

preprocess

t1 = clip.tokenize("Hello World!").cuda()
t2 = clip.tokenize("hi earth").cuda()

text_tokens = [t1, t2]
with torch.no_grad():
    text_features = model.encode_text(text_tokens).float()

image_features /= image_features.norm(dim=-1, keepdim=True)