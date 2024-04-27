import vec2text
import torch
from openai import OpenAI
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
import numpy as np
import os
import time


#array for the sentences we would like to encode
expert_questions = []
expert_talking_points = []
#set up conversation history to prevent repeat questions
questions_converstaion_history = []
talking_points_conversation_history = []


#get the API to produce some questions and some talking points
#questions:
for x in range(0,1):

    content = "Hi expert paleontologist, could you generate 1 question only you could come up with. Please output just the question generated. If this is not the first time being asked this, produce a different question than one generated previously"
    messages = questions_converstaion_history + [
        {"role": "system", "content": "I want you to pretend you are a very knowledgable person and an expert in paleontolgy. you will be prompted to generate some questions that can be answered by only an expert paleontologist"},
        {"role": "user", "content": content}
    ]

    chat_completion = OpenAI.chat.completions.create(
        messages = messages,
        model = "gpt-3.5-turbo",
    )

    chat_response = chat_completion.choices[0].message.content
    print(chat_response)

    expert_questions.append(chat_response)

    questions_converstaion_history.append({"role": "user", "content": content})
    questions_converstaion_history.append({"role": "assistant", "content": chat_response})
    x+=1
#print(expert_questions)

#talking points
# for x in range(0,10):

#     content = "Hi expert paleontologist, could you generate 1 talking point only you could come up with. Please output just the talking point generated. If this is not the first time being asked this, produce a different talking point than one generated previously"
#     messages = talking_points_conversation_history + [
#         {"role": "system", "content": "I want you to pretend you are a very knowledgable person and an expert in paleontolgy. you will be prompted to generate some talking points that only an expert paleontologist would be able to talk about"},
#         {"role": "user", "content": content}
#     ]

#     chat_completion = client.chat.completions.create(
#         messages = messages,
#         model = "gpt-3.5-turbo",
#     )

#     chat_response = chat_completion.choices[0].message.content
#     print(chat_response)

#     expert_talking_points.append(chat_response)

#     talking_points_conversation_history.append({"role": "user", "content": content})
#     talking_points_conversation_history.append({"role": "assistant", "content": chat_response})
#     x+=1
#print(expert_talking_points)


def get_embeddings_openai(text_list, model="text-embedding-ada-002") -> torch.Tensor:
    client = OpenAI()
    response = client.embeddings.create(
        input=text_list,
        model=model,
        encoding_format="float",  # override default base64 encoding...
    )
    #outputs.extend([e["embedding"] for e in response["data"]])
    #idk why but when I was trying to run this line it just didnt work so i had to change it to the
    #for loop below
    #print(response.data[0].embedding)
    #print(response.data[1].embedding)
    outputs = []
    for text in range(0,len(text_list)):
        outputs.append(response.data[text].embedding)
    print(outputs)
    return torch.tensor(outputs)

def create_openai_embeddings():
    #print(expert_questions)
    #for some reason, get_embeddings_openai doesnt like being passed a list of strings that are stored in a variable
    #so passing expert_questions didnt work (im not sure if there is a super easy fix for this)
    embeddings = get_embeddings_openai([
        'What evidence do we have that suggests dinosaurs may have been warm-blooded like modern birds?',
        'What similarities and differences exist in the ways dinosaurs and modern birds reproduce and care for their offspring?'])
    print(embeddings.shape)
    corrector = vec2text.load_pretrained_corrector("text-embedding-ada-002")
    print("Original questions: 'What evidence do we have that suggests dinosaurs may have been warm-blooded like modern birds?', 'What similarities and differences exist in the ways dinosaurs and modern birds reproduce and care for their offspring?'")
    start_time = time.time()
    print("inverted embedding: ", vec2text.invert_embeddings(
        embeddings=embeddings.cuda(),
        corrector=corrector,
        num_steps=20,
        sequence_beam_width=4
    ))
    end_time = time.time()
    print(end_time-start_time, " seconds")


def get_gtr_embeddings(text_list,
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

def create__gtr_embeddings():
    encoder = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
    corrector = vec2text.load_pretrained_corrector("gtr-base")

    embeddings = get_gtr_embeddings(expert_questions, encoder, tokenizer)
    print(embeddings.shape)

    print("Original questions: ", expert_questions)
    start_time = time.time()
    print("inveted embeddings: ", vec2text.invert_embeddings(
        embeddings=embeddings.cuda(),
        corrector=corrector,
        num_steps=20,
        sequence_beam_width=4
    ))
    end_time = time.time()
    print(end_time - start_time, " seconds")


create__gtr_embeddings()

#create_openai_embeddings()

