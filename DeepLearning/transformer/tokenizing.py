import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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