from flask import Flask, render_template, request, jsonify
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

app = Flask(__name__)

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "paste api"
# Your URLs for document loading
urls = ["https://coal.nic.in/acts-rules-policies"]
loaders = UnstructuredURLLoader(urls)
data = loaders.load()
text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(data)
embeddings = OpenAIEmbeddings()
vectorstore_OpenAI = FAISS.from_documents(docs, embeddings)
vectorstore_OpenAI.save_local("faiss_store")

# Load FAISS store and OpenAI LLM
kal = FAISS.load_local("faiss_store", OpenAIEmbeddings())
llm = OpenAI(temperature=0, model_name='text-davinci-003')
chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=kal.as_retriever())


@app.route('/')
def index():
    return render_template('minbot23.html')


@app.route('/ask_question', methods=['POST'])
def ask_question():
    if request.method == 'POST':
        question = request.json['question']
        answer = chain({"question": question}, return_only_outputs=True)
        return jsonify({'answer': answer})


if __name__ == '__main__':
    app.run(debug=True)
