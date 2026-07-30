import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from sentence_transformers import SentenceTransformer

# Apple Silicon(MPS) 사용 가능 여부 확인 후 없으면 CPU로 대체
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {device}")

model_id = "microsoft/Phi-3-mini-4k-instruct"

# trust_remote_code 를 쓰지 않고 transformers 내장 Phi3 구현을 사용한다.
# (허브의 remote code 는 구버전 API 기준이라 최신 transformers 와 충돌함)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,          # MPS 에서는 float16 이 안정적 ("auto" 는 bfloat16 으로 잡힘)
    attn_implementation="sdpa",   # macOS 에는 flash-attention 이 없음
).to(device)

model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = "Write and email apologizing to Sarah for the tragic gardening mishap. Explain how it happened. <|assistant|>"

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("mps")
generation_output = model.generate(input_ids, max_new_tokens=20)

print(tokenizer.decode(generation_output[0]))
print(input_ids)

for id in input_ids[0]:
    print(f"{id}: {tokenizer.decode(id)}")

# Token Embedding
# Tokenizer load
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

# Language Model load
model = AutoModel.from_pretrained("microsoft/deberta-v3-xsmall")

# Tokenizing
tokens = tokenizer("Hello world", return_tensors='pt')

# Caculate Output
output = model(**tokens)[0]
output.shape

for token in tokens['input_ids'][0]:
    print(tokenizer.decode(token))


# Text Embedding
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Text to Embedding Vector
vector = model.encode("Best movie ever!")
vector.shape # 임베딩 벡터의 값 개수 또는 차원은 모델마다 다름