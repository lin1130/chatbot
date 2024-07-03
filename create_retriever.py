# Retrieval Objects: retrival-augmented generation (RAG)
import dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader

review_csv_path = "data/reviews.csv"
review_chroma_path = "chroma_data/"

dotenv.load_dotenv()
reviews_vector_db = Chroma(
    persist_directory=review_chroma_path,
    embedding_function=OpenAIEmbeddings()
)
question = """
Has anyone complained about communication with the hospital staff?
"""
relevant_docs = reviews_vector_db.similarity_search(question, k=3)
output1 = relevant_docs[1].page_content