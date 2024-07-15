# Retrieval Objects: retrival-augmented generation (RAG)
import dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader

review_csv_path = "../data/reviews.csv"
review_chroma_path = "chroma_data/"

dotenv.load_dotenv()
loader = CSVLoader(file_path=review_csv_path, source_column="review")
reviews = loader.load()

reviews_vector_db = Chroma.from_documents(
    reviews,
    OpenAIEmbeddings(),
    persist_directory=review_chroma_path
)

# question = """
# Has anyone complained about communication with the hospital staff?
# """
# relevant_docs = reviews_vector_db.similarity_search(question, k=1)
# output1 = relevant_docs[0].page_content