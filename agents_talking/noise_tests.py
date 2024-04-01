import vec2text
import torch
from openai import OpenAI
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
import numpy as np
import os
import time
from difflib import SequenceMatcher
import pandas as pd
import matplotlib.pyplot as plt

torch.set_printoptions(precision=10)

#Testing how adding noise ruins the sentence
encoder = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to("cuda")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
corrector = vec2text.load_pretrained_corrector("gtr-base")

#Cheeky embedding and inverting function using gtr embedding for now
def embedd_text(text_list,
                       encoder: PreTrainedModel,
                       tokenizer: PreTrainedTokenizer) -> torch.Tensor:

        inputs = tokenizer(text_list,
                       return_tensors="pt",
                       max_length=128,
                       truncation=True,
                       padding="max_length",).to("cuda")

        with torch.no_grad():
            model_output = encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            hidden_state = model_output.last_hidden_state
            embeddings = vec2text.models.model_utils.mean_pool(hidden_state, inputs['attention_mask'])

        return embeddings
    
def invert_embedding(embeddings):
    start_time = time.time()
    inverted_embeddings = vec2text.invert_embeddings(
        embeddings=embeddings.cuda(),
        corrector=corrector,
        num_steps=20,
        sequence_beam_width=4
        )
    end_time = time.time()
    print(end_time - start_time, " seconds to invert embedding")
    return inverted_embeddings


# text_to_embed = ['What multi-disciplinary techniques do paleontologists use to analyze the chemical composition of fossilized bones for insights into ancient environments and dietary habits of extinct species?',
#                  'What evidence do we have to suggest that feathered dinosaurs may have exhibited complex mating behaviors similar to modern birds?']


# ****** so far this code just does the results for a single sentence but ill change it tomorrow *******
text_to_embed = ['What multi-disciplinary techniques do paleontologists use to analyze the chemical composition of fossilized bones for insights into ancient environments and dietary habits of extinct species?']

embeddings = embedd_text(text_to_embed, encoder, tokenizer)
#print(embeddings.shape)
embeddings = embeddings.cpu()
#print(embeddings)
#print(embeddings[0,0])

rounded_embeddings = torch.empty((0,768))
rand_noise_embeddings = torch.empty((0,768))
temp_tensor = embeddings
#print(rounded_embeddings.shape)

rounded_cosine_similarity = []
noise_cosine_similarity = []

#for cosine similarity
cosi = torch.nn.CosineSimilarity()

loops = 10
for i in range(loops,0,-1):
    #calc rounded embeddings
    rounded_embeddings = torch.cat([rounded_embeddings, torch.round(embeddings,decimals = i)], dim = 0)
    rounded_cosine_similarity.append((cosi(embeddings, rounded_embeddings[loops-i])).item())
    #calc rand noise embeddings
    temp_tensor += np.random.rand()/loops**i
    rand_noise_embeddings = torch.cat([rand_noise_embeddings, temp_tensor], dim = 0)
    noise_cosine_similarity.append((cosi(embeddings, rand_noise_embeddings[loops-i])).item())
    temp_tensor = embeddings
# print(rounded_embeddings.shape)
# print(rand_noise_embeddings.shape)
# print(rounded_embeddings)
# print(rand_noise_embeddings)
print("rounded cosine similarity: ", rounded_cosine_similarity)
print("rand noise cosine similarity: ", noise_cosine_similarity)

inverted_rounded_embeddings = invert_embedding(rounded_embeddings)
inverted_rand_noise_embeddings = invert_embedding(rand_noise_embeddings)
print(inverted_rounded_embeddings)
print(inverted_rand_noise_embeddings)

rounded_sentence_similarity = []
noise_sentence_similarity = []
rounded_sentence_similarity_no_spaces = []
noise_sentence_similarity_no_spaces = []

#compute sentence similarity of inverted strings vs original
#for sentences in inverted_rounded_embeddings: (<-- for some reason it didnt like me doing it this way)
for i in range(0,loops):
    rounded_sentence_similarity.append(SequenceMatcher(None, text_to_embed[0], inverted_rounded_embeddings[i]).ratio())
    rounded_sentence_similarity_no_spaces.append(SequenceMatcher(None, text_to_embed[0], inverted_rounded_embeddings[i].replace(" ", "")).ratio())
    noise_sentence_similarity.append(SequenceMatcher(None, text_to_embed[0], inverted_rand_noise_embeddings[i]).ratio())
    noise_sentence_similarity_no_spaces.append(SequenceMatcher(None, text_to_embed[0], inverted_rand_noise_embeddings[i].replace(" ", "")).ratio())

print("Rounded sentence similarities: ", rounded_sentence_similarity)
print("Rounded sentence similarities no spaces: ", rounded_sentence_similarity_no_spaces)
print("Rand Noise sentence similarities: " , noise_sentence_similarity)
print("Rand Noise sentence similarities no spaces: ", noise_sentence_similarity_no_spaces)


#if this comment is still here when you look at my code just know ill change/format the graphs and make them nicer, add titles etc
#i just wanted to test if everything was working and i was lazy cos it was late
df = pd.DataFrame({
    "Rounded sentence similarities": rounded_sentence_similarity,
    "Rounded sentence similarities no spaces": rounded_sentence_similarity_no_spaces,
    "Cosine similarity": rounded_cosine_similarity
    },
    index = range(0,loops)
)
df.plot.line()

df = pd.DataFrame({
    "Rand Noise sentence similarities": noise_sentence_similarity,
    "Rand Noise similarities no spaces": noise_sentence_similarity_no_spaces,
    "Cosine similarity": noise_cosine_similarity
    },
    index = range(0,loops)
)
df.plot.line()
plt.show()
