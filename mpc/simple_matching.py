# pip install crypten

import torch
import numpy as np
import crypten
from torch.nn.functional import normalize
import crypten.mpc as mpc
import crypten.communicator as comm 
import pickle

from crypten.config import cfg
# Initialize crypten
crypten.init()
# Disables OpenMP threads -- needed by @mpc.run_multiprocess which uses fork
torch.set_num_threads(1)

def set_precision(bits):
    cfg.encoder.precision_bits = bits

# This is just a little demo to show that this all works
A = torch.randn(10, 10)
A_n = normalize(A)
A_enc = crypten.cryptensor(A_n) # encrypt
A_dec = A_enc.get_plain_text() # decrypt
input_mse = ((A - A_dec)**2).mean()

B =  torch.randn(100, 10)
B_n = normalize(B).t()
B_enc = crypten.cryptensor(B_n)

threshold = 0.4
threshold_enc = crypten.cryptensor(threshold)

# Cosine similarity between A and B
cosine_sim_raw = A_n @ B_n
selected_indices_raw = cosine_sim_raw >= threshold

# Now let's do it in MPC
cosine_sim_enc = A_enc.matmul(B_enc)
cosine_sim = cosine_sim_enc.get_plain_text()

cosine_sim_enc = A_enc.matmul(B_enc)
selected_indices = cosine_sim_enc >= threshold_enc
selected_indices = selected_indices.get_plain_text()

cosine_mse = ((cosine_sim_raw - cosine_sim)**2).mean()
selected_indices_mse = np.array(selected_indices_raw == selected_indices).mean()


@mpc.run_multiprocess(world_size=2)
def cosine_similarity_mpc(A: torch.Tensor, B: torch.Tensor) -> bytes:
    """
    Computes the cosine similarity between two tensors A and B using crypten. Pre-normalizes for efficiency
    This is secure because the user owns A and can do this locally, and the parties can jointly pre-process norm(B)
    which is a one-time MPC operation
    """

    A_normed = A / torch.sqrt(torch.sum(A * A, dim=1, keepdim=True))
    B_normed = B / torch.sqrt(torch.sum(B * B, dim=1, keepdim=True))

    # secret-share A_normed, B_normed
    A_normed_enc = crypten.cryptensor(A_normed, ptype=crypten.mpc.arithmetic)
    B_normed_enc = crypten.cryptensor(B_normed, ptype=crypten.mpc.arithmetic)

    # Compute the dot product of A and B
    cosine_sim = A_normed_enc.matmul(B_normed_enc)

    # We need to decrypt and convert to binary to return from the MPC
    cosine_sim_binary = pickle.dumps(cosine_sim.get_plain_text())
    return cosine_sim_binary


@mpc.run_multiprocess(world_size=2)
def threshold_sim_mpc(A: torch.Tensor, B: torch.Tensor, threshold: torch.Tensor) -> bytes:
    """
    Finds the boolean matrix of which vectors are above a threshold of cosine similarity closeness to B vectors. Builds off `cosine_similarity_mpc`.
    """

    A_normed = A / torch.sqrt(torch.sum(A * A, dim=1, keepdim=True))
    B_normed = B / torch.sqrt(torch.sum(B * B, dim=1, keepdim=True))

    # secret-share A_normed, B_normed
    A_normed_enc = crypten.cryptensor(A_normed, ptype=crypten.mpc.arithmetic)
    B_normed_enc = crypten.cryptensor(B_normed, ptype=crypten.mpc.arithmetic)
    threshold_enc = crypten.cryptensor(threshold, ptype=crypten.mpc.arithmetic)

    # Compute the dot product of A and B
    cosine_sim = A_normed_enc.matmul(B_normed_enc)

    selected_indices = cosine_sim >= threshold_enc

    # # We need to decrypt and convert to binary to return from the MPC
    selected_indices_binary = pickle.dumps(selected_indices.get_plain_text())
    return selected_indices_binary

results = []
for n in [10, 100, 1000]:
    for repeat in range(3):
        A = torch.randn(n, 10)
        B = torch.randn(n, 10)
        threshold = torch.tensor(0.4)
        cosine_sim = pickle.loads(cosine_similarity_mpc(A, B))
        selected_indices = pickle.loads(threshold_sim_mpc(A, B, threshold))
        cosine_sim_raw = normalize(A) @ normalize(B).t()
        selected_indices_raw = cosine_sim_raw >= threshold
        cosine_sim_mse = ((cosine_sim_raw - cosine_sim)**2).mean()
        selected_indices_mse = np.array(selected_indices_raw == selected_indices).mean()
        results.append((cosine_sim_mse, selected_indices_mse))

