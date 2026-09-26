import numpy as np
import pandas as pd

from datasets import load_dataset

from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

from matplotlib import pyplot as plt

dataset = load_dataset("maartengr/arxiv_nlp")["train"]

abstracts = dataset["Abstracts"]
titles = dataset["Titles"]

embedding_model = SentenceTransformer("thenlper/gte-small")
embeddings = embedding_model.encode(abstracts, show_progress_bar=True)
embeddings.shape
# (44949, 384)

# Dimension Reduction with UMAP
umap_model = UMAP(
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=42
)

reduced_embeddings = umap_model.fit_transform(embeddings)

# Embedding Clustering Reduction
hdbscan_model = HDBSCAN(min_cluster_size=50).fit(reduced_embeddings)
clusters = hdbscan_model.labels_

print(len(set(clusters)))  # 157

cluster = 0
for index in np.where(clusters == cluster)[0][:3]:
    print(abstracts[index][:300] + "... \n")

#   This works aims to design a statistical machine translation from English text
# to American Sign Language (ASL). The system is based on Moses tool with some
# modifications and the results are synthesized through a 3D avatar for
# interpretation. First, we translate the input text to gloss, a written fo...
#
#   Researches on signed languages still strongly dissociate lin- guistic issues
# related on phonological and phonetic aspects, and gesture studies for
# recognition and synthesis purposes. This paper focuses on the imbrication of
# motion and meaning for the analysis, synthesis and evaluation of sign lang...

reduced_embeddings = UMAP(
    n_components=2,
    min_dist=0.0,
    metric="cosine",
    random_state=42
).fit_transform(embeddings)

df = pd.DataFrame(reduced_embeddings, columns=["x", "y"])
df["title"] = titles
df["cluster"] = [str(c) for c in clusters]

df_cluster = df.loc[df.cluster != "-1", :]
df_outliers = df.loc[df.cluster == "-1", :]
# (-9.785034656524658,
#  11.501396656036377,
#  -1.3652766525745392,
#  16.234081465005875)

plt.scatter(df_outliers.x, df_outliers.y, alpha=0.05, s=2, c="grey")
plt.scatter(df_cluster["x"], df_cluster["y"], c=df_cluster.cluster.astype(int), alpha=0.6, s=2, cmap="tab20b")
plt.axis("off")
plt.show()