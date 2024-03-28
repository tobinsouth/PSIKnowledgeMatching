import vec2text
import torch
from openai import OpenAI
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
import numpy as np
import os
import time

client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY")
)

class LLM:
    def __init__(self, name):
        self.name = name
        self.generated_questions = []
        self.questions_converstaion_history = []
        self.generated_answers = []

        #for the embeddings and inverting embeddings
        self.encoder = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
        self.corrector = vec2text.load_pretrained_corrector("gtr-base")

    def generate_questions(self, num_questions):
        for x in range(0, num_questions):

            content = "Hi expert paleontologist, could you generate 1 question only you could come up with. Please output just the question generated. If this is not the first time being asked this, produce a different question than one generated previously"
            messages = self.questions_converstaion_history + [
                {"role": "system", "content": "I want you to pretend you are a very knowledgable person and an expert in paleontolgy. you will be prompted to generate some questions that can be answered by only an expert paleontologist"},
                {"role": "user", "content": content}
            ]

            chat_completion = client.chat.completions.create(
                messages = messages,
                model = "gpt-3.5-turbo",
            )

            chat_response = chat_completion.choices[0].message.content
            #print(chat_response)

            self.generated_questions.append(chat_response)

            self.questions_converstaion_history.append({"role": "user", "content": content})
            self.questions_converstaion_history.append({"role": "assistant", "content": chat_response})
            x+=1
        return self.generated_questions
    
    def generate_answers(self, questions):
        #for now just test on 1 question at a time

        #content = questions
        content = ["What unique adaptations did the dinosaur Spinosaurus have that allowed it to thrive in its aquatic environment?"]
        messages = [
                {"role": "system", "content": "I want you to pretend you are a very knowledgable person and an expert in paleontolgy. You will be asked some questions that can be answered by only an expert paleontologist, please answer them to the best of your ability. Please output only the answer"},
                {"role": "user", "content": content}
            ]

        chat_completion = client.chat.completions.create(
                messages = messages,
                model = "gpt-3.5-turbo",
            )

        chat_response = chat_completion.choices[0].message.content
        self.generated_answers.append(chat_response)
        return self.generate_answers
    
    def print_generated_answers(self):
        print(self.generated_answers)

    def embedd_text(self, text_list,
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
    
    def invert_embedding(self, embeddings):
        start_time = time.time()
        inverted_embeddings = vec2text.invert_embeddings(
            embeddings=embeddings.cuda(),
            corrector=self.corrector,
            num_steps=20,
            sequence_beam_width=4
            )
        end_time = time.time()
        print(end_time - start_time, " seconds to invert embedding")
        return inverted_embeddings

def calc_simularity(embedding1, embedding2):
    #there are faster ways to do this im sure but im lazy and this was the first one i found
    embedding1 = embedding1.cpu()
    embedding2 = embedding2.cpu()
    embedding1 = embedding1.numpy()
    embedding2 = embedding2.numpy()
    return np.dot(embedding1,embedding2)/(np.linalg.norm(embedding1) * np.linalg.norm(embedding2))



LLM1 = LLM("Question_Asker")
LLM2 = LLM("Question_Answerer")

#get the 1st LLM to generate a question
question = LLM1.generate_questions(1)
print(question)

#then embedd it and send it to the other LLM to invert
embedded_question = LLM1.embedd_text(question, LLM1.encoder, LLM1.tokenizer)
print(embedded_question)
# print(embedded_question.shape)
inverted_embedded_question = LLM2.invert_embedding(embedded_question)
print(inverted_embedded_question)

#getting an error for some reason need to fix
#Get the other LLM to answer the question, embedd it and send it back to the first LLM
answer_to_embedded_question = LLM2.generate_answers(inverted_embedded_question)
print(answer_to_embedded_question)
embedded_answer = LLM2.embedd_text(answer_to_embedded_question, LLM2.encoder, LLM2.tokenizer)
print(embedded_answer)
# inverted_embedded_answer = LLM1.invert_embedding(embedded_answer)
# print(inverted_embedded_answer)

print("similarity = ", calc_simularity(embedded_question,embedded_answer))
