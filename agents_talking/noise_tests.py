import vec2text
import torch
from openai import OpenAI
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
import numpy as np
import os
import time


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


text_to_embed = ['What multi-disciplinary techniques do paleontologists use to analyze the chemical composition of fossilized bones for insights into ancient environments and dietary habits of extinct species?']


embeddings = embedd_text(text_to_embed, encoder, tokenizer)
print(embeddings.shape)
embeddings = embeddings.cpu()
print(embeddings)

rounded_embeddings = torch.empty((0,768))
rand_noise_embeddings = torch.empty((0,768))
temp_tensor = embeddings
#print(rounded_embeddings.shape)

for i in range(10,0,-1):
     rounded_embeddings = torch.cat([rounded_embeddings, torch.round(embeddings,decimals = i)], dim = 0)
     temp_tensor += np.random.rand()/10**i
     #rand_noise_embeddings = torch.cat([rand_noise_embeddings, (embeddings[0,:] + np.random.rand()/10**i)], dim = 0)
     rand_noise_embeddings = torch.cat([rand_noise_embeddings, temp_tensor], dim = 0)
     temp_tensor = embeddings
print(rounded_embeddings.shape)
print(rand_noise_embeddings.shape)
print(rounded_embeddings)
print(rand_noise_embeddings)

#print(invert_embedding(rounded_embeddings))
print(invert_embedding(rand_noise_embeddings))

