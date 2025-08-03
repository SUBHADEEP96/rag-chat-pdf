
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
import os

load_dotenv()  
openai_api_key = os.getenv("OPENAI_API_KEY")



loader = PyPDFLoader("./Subhadeep_Paul.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Vectorize
embeddings = OpenAIEmbeddings(model="text-embedding-3-large",openai_api_key=openai_api_key)

vectorstore = FAISS.from_documents(chunks, embeddings)

# Set up LLM and chain

llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key,model="gpt-4o")

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
)

response = chain.invoke({"question": "how much experience subhadeep has in gen AI?","chat_history": []} )

print("Response:", response['answer'])


