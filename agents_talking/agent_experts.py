import os
import sys
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from torch import embedding_bag

model = SentenceTransformer("all-MiniLM-L6-v2")

client = OpenAI()

#array for the sentences we would like to encode
expert_questions = []
expert_talking_points = []

#get the API to produce some questions and some talking points

#set up conversation history to prevent repeat questions
questions_converstaion_history = []
talking_points_conversation_history = []

#get api to spit some stuff out
#questions:
for x in range(0,1):

    content = "Hi expert paleontologist, could you generate 1 question only you could come up with. Please output just the question generated. If this is not the first time being asked this, produce a different question than one generated previously"
    messages = questions_converstaion_history + [
        {"role": "system", "content": "I want you to pretend you are a very knowledgable person and an expert in paleontolgy. you will be prompted to generate some questions that can be answered by only an expert paleontologist"},
        {"role": "user", "content": content}
    ]

    chat_completion = client.chat.completions.create(
        messages = messages,
        model = "gpt-3.5-turbo",
    )

    chat_response = chat_completion.choices[0].message.content
    print(chat_response)

    expert_questions.append(chat_response)

    questions_converstaion_history.append({"role": "user", "content": content})
    questions_converstaion_history.append({"role": "assistant", "content": chat_response})
    x = x + 1
print(expert_questions)

#talking points
for x in range(0,10):

    content = "Hi expert paleontologist, could you generate 1 talking point only you could come up with. Please output just the talking point generated. If this is not the first time being asked this, produce a different talking point than one generated previously"
    messages = talking_points_conversation_history + [
        {"role": "system", "content": "I want you to pretend you are a very knowledgable person and an expert in paleontolgy. you will be prompted to generate some talking points that only an expert paleontologist would be able to talk about"},
        {"role": "user", "content": content}
    ]

    chat_completion = client.chat.completions.create(
        messages = messages,
        model = "gpt-3.5-turbo",
    )

    chat_response = chat_completion.choices[0].message.content
    print(chat_response)

    expert_talking_points.append(chat_response)

    talking_points_conversation_history.append({"role": "user", "content": content})
    talking_points_conversation_history.append({"role": "assistant", "content": chat_response})
    x = x + 1
print(expert_talking_points)



#lets encode the sentences
expert_questions_embeddings = model.encode(expert_questions)
expert_talking_points_embeddings = model.encode(expert_talking_points)

#print these suckas
for sentence, embedding in zip(expert_questions, expert_questions_embeddings):
    print("sentence:", sentence)
    print("embedding:", embedding)
    # with open('e:/VisualStudioCodeDocuments/Internship stuff/PSIKnowledgeMatching/agents_talking/embedded_questions.txt', 'w') as f:
    #     print(sentence, repr(embedding), file=f)
    print("")

for sentence, embedding in zip(expert_talking_points, expert_talking_points_embeddings):
    print("sentence:", sentence)
    print("embedding:", embedding)
    print("")








    