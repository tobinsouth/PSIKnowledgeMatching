
# This is the file we're we can show how to use PSI to compare the embeddings of the documents and queries.

import private_set_intersection.python as psi # !pip install openmined.psi
import os, json, random, pickle
from beir.datasets.data_loader import GenericDataLoader
import numpy as np

corpus, queries, qrels = GenericDataLoader(data_folder='datasets/subquora').load(split="test")

with open("datasets/subquora/corpus_embeddings.pkl", "rb") as f:
    corpus_embeddings_dict = pickle.load(f)

with open("datasets/subquora/query_embeddings.pkl", "rb") as f:
    query_embeddings_dict = pickle.load(f)

X = np.stack(list(corpus_embeddings_dict.values())+list(query_embeddings_dict.values()))


doc_embedding = query_embeddings_dict['443118']

np.floor(doc_embedding.round(0))


# plot pca of x
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.show()


fpr = 0.01
num_client_inputs = 10
num_server_inputs = 100
client_items = ["Element " + str(i) for i in range(num_client_inputs)]
server_items = ["Element " + str(2 * i) for i in range(num_server_inputs)]
reveal_intersection = True
MAX_CLIENT_ITEMS = 1000


client_key = bytes(range(32))
server_key = bytes(range(1, 33))
c = psi.client.CreateWithNewKey(reveal_intersection)
s = psi.server.CreateWithNewKey(reveal_intersection)


def setup_server(server_items, fpr=0.01, ds=psi.DataStructure.GCS):
    # ds = psi.DataStructure.GCS # OR psi.DataStructure.BLOOM_FILTER
    setup = psi.ServerSetup()
    setup.ParseFromString(
        s.CreateSetupMessage(fpr, MAX_CLIENT_ITEMS, server_items, ds).SerializeToString()
    )
    return setup

def check_intersection(setup, client_items):
    client_items = client_items[:MAX_CLIENT_ITEMS] + ["" for _ in range(MAX_CLIENT_ITEMS - len(client_items))]

    request = psi.Request()
    request.ParseFromString(c.CreateRequest(client_items).SerializeToString())

    # this would then be sent to the server, s

    response = psi.Response()
    response.ParseFromString(s.ProcessRequest(request).SerializeToString())

    intersection = c.GetIntersection(setup, response)

    return intersection


setup = setup_server(server_items)
intersection = check_intersection(setup, client_items)
