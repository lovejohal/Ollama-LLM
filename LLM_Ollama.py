from flask import Flask, request, jsonify
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
app = Flask(__name__)

# Load local documents
documents = SimpleDirectoryReader("data").load_data()

# Initialize the embeddings model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# Initialize the language model
Settings.llm = Ollama(model="qwen2:0.5b", request_timeout=360.0)

# Create the index with the embeddings and language model
index = VectorStoreIndex.from_documents(documents,)

# Create the query engine
query_engine = index.as_query_engine()

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query_text = data.get('query', '')
    if not query_text:
        return jsonify({'error': 'Query text is required'}), 400

    try:
        response = query_engine.query(query_text)
        return jsonify({'response': str(response)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test', methods=['GET'])
def test():
    
    try:
        print("klhjkh")
        return jsonify({'response': "nmdfgb"})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001)
