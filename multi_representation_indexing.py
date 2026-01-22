import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from langchain_core.stores import InMemoryByteStore
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
import uuid
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()
from langchain.embeddings.base import Embeddings

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class BertEmbeddings(Embeddings):
    def __init__(self,model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy = True,
                                 normalize_embeddings = True).tolist()
    def embed_query(self, text):
        return self.model.encode(
                [text],
                convert_to_numpy = True,
                normalize_embeddings = True,
                )[0].tolist()

chat = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
from pathlib import Path
file_path = Path(__file__).parent / "langsmith.txt"

loaders = [
    TextLoader(
        file_path = str(file_path),
        encoding = "utf-8"
        ),
    
]

documents = []
for item in loaders:
    documents.extend(item.load())

splitter = RecursiveCharacterTextSplitter(chunk_size = 10000)
documents = splitter.split_documents(documents)

chain = (
    {"doc": lambda x: x.page_content} 
    | ChatPromptTemplate.from_template("Summarize the following document: \n\n{doc}")
    |chat
    |StrOutputParser()
    
    )

summaries = chain.batch(documents, {"max_concurrency" : 3})

#Index with multi-representations

embedding_model = SentenceTransformer("bert-base-uncased")
vectorstore = Chroma(collection_name="summaries", embedding_function = BertEmbeddings(embedding_model))
store = InMemoryByteStore()
id_key = "doc_id"

retriever = MultiVectorRetriever(
        vectorstore = vectorstore,
        byte_store = store,
        id_key = id_key,
        )

doc_ids = [str(uuid.uuid4()) for _ in documents]

summary_docs = [
        Document(page_content = s, metadata = {id_key: doc_ids[i]})
        for i, s in enumerate(summaries)
        ]

retriever.vectorstore.add_documents(summary_docs)
retriever.docstore.mset(list(zip(doc_ids, documents)))


query = "What is LangSmith?"
sub_docs = vectorstore.similarity_search(query)
sub_docs[0]

retrieved_docs = retriever.invoke(query)

retrieved_docs[0].page_content[0:500]

print(len(retrieved_docs[0].page_content))
