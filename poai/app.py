from flask import Flask, request, jsonify, render_template
import os
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini  
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage

app = Flask(__name__)
UPLOAD_FOLDER = './doc'
PERSIST_DIR = './storage'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

os.environ["GOOGLE_API_KEY"] = "AIzaSyAN8TYrIPfDYZU-wZ8L645gjSew5J4_IIA"  # Add your API key
gemini_embedding_model = GeminiEmbedding(model_name="models/embedding-001")
llm = Gemini()
Settings.llm = llm
Settings.embed_model = gemini_embedding_model
Settings.num_output = 2080
Settings.context_window = 3900

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Process document
    documents = SimpleDirectoryReader(UPLOAD_FOLDER).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

    return jsonify({"message": "File uploaded and processed successfully"})

@app.route('/ask', methods=['POST'])
def ask():
    question = request.json.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400

    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context=storage_context)
    query_engine = index.as_query_engine()

    response = query_engine.query(question)
    return jsonify({"response": response.response})

if __name__ == '__main__':
    app.run(debug=True)
