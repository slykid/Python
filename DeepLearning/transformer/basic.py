import torch
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

# 문장에 대한 감정분석
classifier("I've been waiting for a HuggingFace course my whole life.")

classifier(
    ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]
)
# [{'label': 'POSITIVE', 'score': 0.9598051905632019},
#  {'label': 'NEGATIVE', 'score': 0.9994558691978455}]

# 제로샷 분류 (Zero-shot Classification)
classifier = pipeline("zero-shot-classification")
classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)
# {'sequence': 'This is a course about the Transformers library',
#  'labels': ['education', 'business', 'politics'],
#  'scores': [0.8445976972579956, 0.11197519302368164, 0.043427132070064545]}

# 텍스트 생성 (Text Generation)
generator = pipeline("text-generation")
generator("In this course, we will teach you how to")
# [{'generated_text': 'In this course, we will teach you how to create, manage and manage your own team.
#       Each program will include an Introduction to the Company course, a Practicum of Leadership,
#       five course exercises and one video. It will also be useful for businesses'}]

# 마스크 채우기
unmasker = pipeline("fill-mask")
unmasker("This course will teach you all about <mask> models.", top_k=2)
# [{'score': 0.19620023667812347,
#   'token': 30412,
#   'token_str': ' mathematical',
#   'sequence': 'This course will teach you all about mathematical models.'},
#  {'score': 0.04052722454071045,
#   'token': 38163,
#   'token_str': ' computational',
#   'sequence': 'This course will teach you all about computational models.'}]

# 개체명 인식
ner = pipeline("ner", grouped_entities=True)
ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
# [{'entity_group': 'PER',
#   'score': 0.9981694,
#   'word': 'Sylvain',
#   'start': 11,
#   'end': 18},
#  {'entity_group': 'ORG',
#   'score': 0.9796019,
#   'word': 'Hugging Face',
#   'start': 33,
#   'end': 45},
#  {'entity_group': 'LOC',
#   'score': 0.9932106,
#   'word': 'Brooklyn',
#   'start': 49,
#   'end': 57}]

# 질의응답
question_answerer = pipeline("question-answering")
question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)
# {'score': 0.6949762105941772, 'start': 33, 'end': 45, 'answer': 'Hugging Face'}

# 요약
summarizer = pipeline("summarization")
summarizer(
    """
    America has changed dramatically during recent years. Not only has the number of
    graduates in traditional engineering disciplines such as mechanical, civil,
    electrical, chemical, and aeronautical engineering declined, but in most of
    the premier American universities engineering curricula now concentrate on
    and encourage largely the study of engineering science. As a result, there
    are declining offerings in engineering subjects dealing with infrastructure,
    the environment, and related issues, and greater concentration on high
    technology subjects, largely supporting increasingly complex scientific
    developments. While the latter is important, it should not be at the expense
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other
    industrial countries in Europe and Asia, continue to encourage and advance
    the teaching of engineering. Both China and India, respectively, graduate
    six and eight times as many traditional engineers as does the United States.
    Other industrial countries at minimum maintain their output, while America
    suffers an increasingly serious decline in the number of engineering graduates
    and a lack of well-educated engineers.
"""
)
# [{'summary_text': ' The number of engineering graduates in the United States has declined in recent years.
#           China and India graduate six and eight times as many traditional engineers as the U.S. does.
#           Rapidly developing economies such as China continue to encourage and advance the teaching of engineering.
#           There are declining offerings in engineering subjects dealing with infrastructure, infrastructure,
#           the environment, and related issues.'
# }]

