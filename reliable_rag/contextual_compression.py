from dotenv import load_dotenv
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma

from helper_functions import *
from evaluate_rag import *

# Load environment variables from a .env file
load_dotenv()


data_path = {'path': "data/test/", 'glob': '*.txt'}
loader = DirectoryLoader(data_path['path'], glob=data_path['glob'], show_progress=True, use_multithreading=False)
documents = loader.load()
# Split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,  # Increase chunk size to include more context
    chunk_overlap=500,  # Maintain some overlap to ensure continuity
    length_function=len,
    add_start_index=True,
)

splits = text_splitter.split_documents(documents)

# Add to vectorstore
# from sentence_transformers import SentenceTransformer
# embedding_model = SentenceTransformer('emilyalsent/bio-sent2vec')
embedding_model = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=splits,
    collection_name="rag",
    embedding=embedding_model,
)

retriever = vectorstore.as_retriever(
    search_type='similarity',  # Maximal Marginal Relevance (MMR)
    search_kwargs={'k': 25},  # retrieve more chunks
)

#Create a contextual compressor
# llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=4000)
llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
compressor = LLMChainExtractor.from_llm(llm)

#Combine the retriever with the compressor
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

# Create a QA chain with the compressed retriever
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=compression_retriever,
    return_source_documents=True
)

from data.prompt import question
result = qa_chain.invoke({"query": question})
print(result["result"])
print("Source documents:", result["source_documents"])
