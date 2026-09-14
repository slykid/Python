import numpy as np
from tqdm import tqdm

from datasets import load_dataset
from sklearn.metrics import classification_report

from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

def evaluate_performance(y_true, y_pred):
    """분류 레포트를 만들어 출력합니다."""
    performance = classification_report(
        y_true, y_pred,
        target_names=["Negative Review", "Positive Review"]
    )

    print(performance)

# 데이터 로드
data = load_dataset("cornell-movie-review-data/rotten_tomatoes")
data
# DatasetDict({
#     train: Dataset({
#         features: ['text', 'label'],
#         num_rows: 8530
#     })
#     validation: Dataset({
#         features: ['text', 'label'],
#         num_rows: 1066
#     })
#     test: Dataset({
#         features: ['text', 'label'],
#         num_rows: 1066
#     })
# })

data["train"][0, -1]
# {'text': ['the rock is destined to be the 21st century\'s new " conan " and that he\'s going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .',
#   'things really get weird , though not particularly scary : the movie is all portent and no content .'],
#  'label': [1, 0]}

# 허깅페이스 모델 경로
model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# 파이프라인으로 모델 로드
pipe = pipeline(
    model=model_path,
    tokenizer=model_path,
    return_all_scores=True,
    device="mps"
)


y_pred = []

# 파이프라인 호출 시 인자로 top_k=None을 명시적으로 전달합니다.
for output in tqdm(pipe(KeyDataset(data["test"], "text"), top_k=None), total=len(data["test"])):

    # 1. 반환 형태 강제 정규화 (딕셔너리 하나만 나오거나 이중 리스트로 나올 경우 대비)
    if isinstance(output, dict):
        output = [output]
    elif isinstance(output, list) and len(output) > 0 and isinstance(output[0], list):
        output = output[0]

    # 2. label을 Key로 하는 딕셔너리로 변환
    score_dict = {item["label"]: item["score"] for item in output}

    # 3. .get()을 사용해 에러를 방지하고, 값을 못 찾으면 0.0을 부여
    negative_score = score_dict.get("negative", 0.0)
    positive_score = score_dict.get("positive", 0.0)

    # 4. 부정(0)과 긍정(1) 점수 비교
    assignment = np.argmax([negative_score, positive_score])
    y_pred.append(assignment)

evaluate_performance(data["test"]["label"], y_pred)
#                    precision    recall  f1-score   support
# Negative Review       0.76      0.88      0.81       533
# Positive Review       0.86      0.72      0.78       533
#
# accuracy                               0.80      1066
# macro avg          0.81      0.80      0.80      1066
# weighted avg       0.81      0.80      0.80      1066
