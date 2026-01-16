from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
from langchain_core.prompts import PromptTemplate


class BertEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(
                texts,
                convert_to_numpy  = True,
                normalize_embeddings = True,
                ).tolist()
    def embed_query(self,text):
        return self.model.encode(
                [text],
                convert_to_numpy = True,
                normalize_embeddings = True
                )[0].tolist()


print("=========Loading Secret Token=========\n")
load_dotenv()

print("=========Loading Model HF way=========\n")
'''
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device, "\n")
#model = model.to(device)
model.eval()

inputs = tokenizer(
        text,
        return_tensors = "pt",
        truncation = True,
        padding = True
        )

with torch.no_grad():
    outputs = model(**inputs)

print("=========Generate embeddings and pooling manually for SE=========\n")
#mean pooling
token_embeddings = outputs.last_hidden_state
attention_mask = inputs["attention_mask"]
mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
sum_embeddings = torch.sum(token_embeddings * mask, dim =1)
sum_mask = torch.clamp(mask.sum(dim =1), min = 1e-9)
embeddings = sum_embeddings/ sum_mask
print(embeddings.shape)
print(len(embeddings))
print(len(embeddings[0]))
'''


#Simple way to generate embeddings: 
text = [
    "This is the Fundamentals of RAG course.",
    "Educative is an AI-powered online learning platform.",
    "There are several Generative AI courses available on Educative.",
    "I am writing this using my keyboard.",
    "JavaScript is a good programming language"
    ]


print("=========Loading Model Easy way=========\n")
model = SentenceTransformer("bert-base-uncased")

print("=========Generating Embedding=========")
embeddings = model.encode(
    text,
    convert_to_numpy = True,
    normalize_embeddings = True
    )
print(embeddings.shape)
print(len(embeddings))
print(len(embeddings[0]))


db = Chroma.from_texts(text, BertEmbeddings(model))
print(db)

retriever = db.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k":1}
        )
result = retriever.invoke("What is Educative")
print(result)


template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say 'thanks for asking!' at the end of the answer.

{context}
Question: {question}

Helpful Answer:"""

custom_rag_prompt = PromptTemplate.from_template(template)
#print(custom_rag_prompt)












