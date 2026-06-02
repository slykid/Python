import torch
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

# 문장에 대한 감정분석
classifier("I've been waiting for a HuggingFace course my whole life.")

classifier(
    ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]
)