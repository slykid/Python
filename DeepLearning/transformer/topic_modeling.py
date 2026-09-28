import numpy as np
import pandas as pd

from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from umap import UMAP
from hdbscan import HDBSCAN

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance

from copy import deepcopy

dataset = load_dataset("maartengr/arxiv_nlp")["train"]

abstracts = dataset["Abstracts"]
titles = dataset["Titles"]

embedding_model = SentenceTransformer("thenlper/gte-small")
embeddings = embedding_model.encode(abstracts, show_progress_bar=True)

umap_model = UMAP(n_components=5, min_dist=0.0, metric="cosine", random_state=42)
reduced_embeddings = umap_model.fit_transform(embeddings)

hdbscan_model = HDBSCAN(min_cluster_size=50).fit(reduced_embeddings)

# 토픽 모델링
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    verbose=True
).fit(abstracts, embeddings)

topic_model.get_topic_info()  # 발견한 토픽에 대한 간단한 정보 제공
# Topic  ...                                Representative_Docs
# 0       -1  ...  [  Sentence semantic understanding is a key to...
# 1        0  ...  [  Question generation (QG) attempts to solve ...
# 2        1  ...  [  Recently, masked prediction pre-training ha...
# 3        2  ...  [  Document-level machine translation incorpor...
# 4        3  ...  [  Abstractive summarization systems generally...
# ..     ...  ...                                                ...
# 152    151  ...  [  Sentence representation at the semantic lev...
# 153    152  ...  [  Text generation is of particular interest i...
# 154    153  ...  [  Out-of-distribution (OOD) detection is esse...
# 155    154  ...  [  Prompt optimization aims to find the best p...
# 156    155  ...  [  Diffusion models have achieved great succes...

topic_model.get_topic(0)
# [('question', 0.020918147679363355),
#  ('answer', 0.015676702884762875),
#  ('questions', 0.015625327510168485),
#  ('qa', 0.015625097782508292),
#  ('answering', 0.014495461559385222),
#  ('answers', 0.009752876221658564),
#  ('retrieval', 0.00939374456284612),
#  ('comprehension', 0.007745696187803967),
#  ('reading', 0.007202916376435195),
#  ('the', 0.006151424133013237)]

topic_model.find_topics("topic modeling")
# ([24, -1, 56, 34, 77],
#  [0.9545109983057234,
#   0.9120019607687477,
#   0.905891575738071,
#   0.9053679086852346,
#   0.9033118179968254])

topic_model.get_topic(24)
# [('topic', 0.06808673225288067),
#  ('topics', 0.036131417709942534),
#  ('lda', 0.016827433217851417),
#  ('documents', 0.013208306984900752),
#  ('document', 0.01314406901563542),
#  ('latent', 0.013099254157477536),
#  ('modeling', 0.012148092274570064),
#  ('dirichlet', 0.009891090609600198),
#  ('word', 0.00864539374847923),
#  ('allocation', 0.007731079913384699)]

topic_model.topics_[titles.index("BERTopic: Neural topic modeling with a class-based TF-IDF procedure")]
# 24

# Visualize Topic & Docs
fig = topic_model.visualize_documents(titles, reduced_embeddings=reduced_embeddings, width=1200, hide_annotations=True)
fig.update_layout(font=dict(size=16))

topic_model.visualize_barchart()
topic_model.visualize_heatmap()
topic_model.visualize_hierarchy()

original_topics = deepcopy(topic_model.topic_representations_)

def topic_difference(model, original_topics, nr_topics=5):
    df = pd.DataFrame(columns=["Topic", "Originals", "Updates"])

    for topic in range(nr_topics):
        og_words = " | ".join(list(zip(*original_topics[topic]))[0][:5])
        new_words = " | ".join(list(zip(*model.get_topic(topic)))[0][:5])

        df.loc[len(df)] = [topic, og_words, new_words]

    return df

## KeyBERTInspired
representation_model = KeyBERTInspired()
topic_model.update_topics(abstracts, representation_model=representation_model)

topic_difference(topic_model, original_topics)
# Topic  ...                                            Updates
# 0      0  ...  answering | questions | comprehension | questi...
# 1      1  ...  encoder | phonetic | transcription | language ...
# 2      2  ...  translating | translation | monolingual | tran...
# 3      3  ...  summarization | summarizers | summaries | summ...
# 4      4  ...  hate | hateful | language | languages | cyberb...
# [5 rows x 3 columns]

## MMR
representation_model = MaximalMarginalRelevance()
topic_model.update_topics(abstracts, representation_model=representation_model)
topic_difference(topic_model, original_topics)
# Topic  ...                                            Updates
# 0      0  ...  answering | retrieval | comprehension | hop | ...
# 1      1  ...              speech | asr | wer | audio | training
# 2      2  ...  translation | nmt | neural | parallel | multil...
# 3      3  ...  summarization | summaries | abstractive | docu...
# 4      4  ...         hate | toxic | abusive | hateful | dataset
# [5 rows x 3 columns]

result = topic_difference(topic_model, original_topics)
print(result["Topic"])
print(result["Originals"])
print(result["Updates"])


