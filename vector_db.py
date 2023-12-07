from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.indexes.vectorstore import VectorstoreIndexCreator

def init_db(file_name: str):
    with open(file_name) as f:
        text = f.read()
    docs = embeddings(text, 500)
    return docs

## embedding 완료된 chroma DB를 리턴
def embeddings(text: str, chunk_size: int):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    splitted_text = splitter.split_text(text)

    embeddings = OpenAIEmbeddings()

    doc_search = Chroma.from_texts(
        splitted_text, embeddings, metadatas=[{"source": str(i)} for i in range(len(splitted_text))]
    ).as_retriever()

    return doc_search



