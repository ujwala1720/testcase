from flask import Flask, render_template, request, jsonify
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

app = Flask(_name_)

@app.route('/')
def index():
    return render_template('main_page.html')
@app.route('/main')
def second():
    return render_template('botinterface.html')

@app.route('/process_input', methods=['POST'])
def process_input():
    z = request.json.get('user_input', '')

    os.environ["OPENAI_API_KEY"] = "paste api key"
    urls = ["https://coal.nic.in", "https://mines.gov.in"]
    loaders = UnstructuredURLLoader(urls)
    data = loaders.load()
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings()
    vectorstore_OpenAI = FAISS.from_documents(docs, embeddings)
    vectorstore_OpenAI.save_local("faiss_store")
    kal = FAISS.load_local("faiss_store", OpenAIEmbeddings())
    llm = OpenAI(temperature=0, model_name='text-davinci-003')
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=kal.as_retriever())
    ans = chain({"question": z}, return_only_outputs=True)
    response = ans['answer']
    return jsonify({'response': response})

if _name_ == '_main_':
    app.run(debug=True)
