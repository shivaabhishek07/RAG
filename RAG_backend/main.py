from flask import Flask, request, jsonify
import os
import PyPDF2
from retriever import train_RAG, getAnswer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Start ngrok when app is run

# A simple in-memory "database" to store the document text
TRAINED_TEXT = ""

@app.route('/train', methods=['POST'])
def train():
    global TRAINED_TEXT
    if 'document' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400
    
    file = request.files['document']
    
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"}), 400
    
    if file and file.filename.endswith('.pdf'):
        try:
            # Extract text from the PDF
            reader = PyPDF2.PdfReader(file)
            TRAINED_TEXT = ""
            for page in reader.pages:
                TRAINED_TEXT += page.extract_text()
            print(TRAINED_TEXT)
            # Specify the filename
            filename = "extracted_text.txt"

            # Open a file in write mode and save the TRAINED_TEXT content
            with open(filename, "w") as file:
                file.write(TRAINED_TEXT)
            
            train_RAG()

            
            return jsonify({"success": True, "message": "Training successful"}), 200
        
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
    
    return jsonify({"success": False, "error": "Invalid file type"}), 400

@app.route('/ask', methods=['POST'])
def ask():
    global TRAINED_TEXT
    if not TRAINED_TEXT:
        return jsonify({"answer": "No training data available. Please train the model first."}), 400
    
    data = request.get_json()
    question = data.get('question', '').lower()
    
    if not question:
        return jsonify({"answer": "Please provide a question."}), 400
    
    response_text = getAnswer(question)
    print(response_text)
    
    return jsonify({"answer": response_text}), 200

if __name__ == '__main__':
    app.run()

