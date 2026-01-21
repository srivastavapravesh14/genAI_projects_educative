############################################################
'''
This file demonstrates different chunking strategies used in advanced RAG systems
Chunking strategies:
    1. Fixed-size overlapping sliding window
    2. Recursive structure-aware splitting
    3. Content-aware splitting
    4. Chunking through NLTHTextSplitter from LangChain
'''

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, MarkdownTextSplitter, PythonCodeTextSplitter
import nltk
from typing import List
nltk.download('punkt', quiet = True)

from langchain_text_splitters import NLTKTextSplitter


def fixed_size_OSW(
        content: str, 
        chunk_size: int, 
        chunk_overlap: int
        ) -> List[str]:
    '''
    This function helps in divding text into equally sized chunks based on the count of characters present in the document. To avoid sentences cutting in half, overlapping chunks are used.

    '''
    sentences = content.split("\n")
    text_splitter = CharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    full_chunks = []
    for sentence in sentences:
        chunks = text_splitter.create_documents([sentence])
        full_chunks.extend(chunks)

    return full_chunks


def recursive_splitter(content: str, chunk_size, chunk_overlap) -> List[str]:
    

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)

    full_chunks = text_splitter.create_documents([content])

    return full_chunks

def content_aware_splitting(content, chunk_size, chunk_overlap,markdown = False,python_code = False):
    
    md_chunks = []
    py_chunks = []
    if markdown:
        splitter = MarkdownTextSplitter(chunk_size, chunk_overlap)
        md_chunks.append(splitter.create_documents(content))
    if python_code:
        pysplitter = PythonCodetextSplitter(chunk_size,chunk_overlap)
        py_chunks.append(pysplitter.create_documents(content))
    

def nltk_splitter(content):
    splitter = NLTKTextSplitter()

    chunks = splitter.split_text(text)

    return chunks


def main(text):
    fs_chunks = fixed_size_OSW(text, 128, 20)
    print(f"Output from fixedsize overlapping sliding window splitting:\n {fs_chunks[0].page_content}")
    rs_chunks = recursive_splitter(text,128,20)
    print(f"\n\nOutput from recursive splitting:\n {rs_chunks[0].page_content}")
    nltk_split = nltk_splitter(text)
    print(f"\n\nOutput from NLTK splitting:\n {nltk_split}")

if __name__ == "__main__":
    text = '''
    Chunking is the process of dividing large documents into smaller, meaningful pieces so they can be efficiently processed by language models. One common approach is fixed-size chunking, where text is split into equal-length segments based on a fixed number of characters or tokens. This method is simple and predictable, making it easy to implement, but it often ignores sentence or paragraph boundaries. As a result, important contextual relationships may be broken, which can reduce the quality of downstream tasks such as semantic search or question answering.

Another widely used approach is overlapping chunking, which improves upon fixed-size chunking by allowing consecutive chunks to share a portion of text. The overlap helps preserve context across chunk boundaries, ensuring that important information is not lost when a sentence or idea spans multiple chunks. While overlapping increases storage and computation costs, it significantly improves retrieval accuracy and is commonly used in Retrieval-Augmented Generation (RAG) systems.

Sentence-based chunking focuses on splitting text at sentence boundaries using punctuation or sentence tokenizers. This strategy preserves semantic meaning better than fixed-size chunking because each chunk contains complete thoughts. However, sentence lengths can vary greatly, which may lead to chunks that are too short or too long for certain models. Sentence-based chunking works best when the document contains well-structured prose.

Paragraph-based chunking divides text according to natural paragraph breaks. This approach maintains a higher-level semantic structure by grouping related sentences together. Paragraph chunking is particularly useful for documents such as articles, research papers, and manuals, where each paragraph represents a distinct idea. The downside is that very long paragraphs may still exceed model limits and require additional splitting.

A more advanced technique is semantic chunking, which groups text based on meaning rather than length or formatting. This method uses embeddings or similarity scores to identify logical breakpoints in the text, ensuring that each chunk represents a coherent concept. Although semantic chunking produces high-quality chunks, it is computationally expensive and more complex to implement, making it suitable for high-accuracy systems rather than lightweight pipelines.

Finally, hierarchical chunking combines multiple strategies by first splitting text into large units such as sections or chapters and then further dividing them into smaller chunks. This approach preserves both global structure and local context, enabling more flexible and accurate retrieval. Hierarchical chunking is especially effective for large documents like books, legal texts, or technical documentation where structure plays a critical role.

'''


    main(text)



