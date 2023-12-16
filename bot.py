import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
z="what is mining?"#append the data into this string
os.environ["OPENAI_API_KEY"] = "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"
urls =["https://coal.nic.in","https://mines.gov.in"]
loaders = UnstructuredURLLoader(urls)
data = loaders.load()
text_splitter = CharacterTextSplitter(separator='\n',chunk_size=1000,chunk_overlap=200)
docs = text_splitter.split_documents(data)
embeddings = OpenAIEmbeddings()
vectorstore_OpenAI = FAISS.from_documents(docs,embeddings)
vectorstore_OpenAI.save_local("faiss_store")
kal =FAISS.load_local("faiss_store", OpenAIEmbeddings())
llm=OpenAI(temperature=0, model_name='text-davinci-003')
chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=kal.as_retriever())

ans=chain({"question": z}, return_only_outputs=True)
# the response statement would be "ans['answer']"
