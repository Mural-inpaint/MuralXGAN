import torch
from sentence_transformers import SentenceTransformer, util
from torch.xpu import device

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = SentenceTransformer('clip-ViT-B-32').to(device)

def feature_map(caption: str):
    global MODEL

    with torch.no_grad():
        text_feat = MODEL.encode(caption, convert_to_tensor=True)
        text_feat = text_feat/text_feat.norm(dim=-1, keepdim=True)

    return text_feat

# test
# caption="this image is highlighted in red overlay to indicate the affected areas."
# text_feat = feature_map(caption)
# print(text_feat.shape, text_feat)