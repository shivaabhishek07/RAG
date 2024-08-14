import React, { useState } from 'react';
import axios from 'axios';

const RagWebsite = () => {
    const [file, setFile] = useState(null);
    const [isTrained, setIsTrained] = useState(false);
    const [question, setQuestion] = useState('');
    const [answer, setAnswer] = useState('');

    // Function to handle file upload
    const handleFileUpload = (e) => {
        setFile(e.target.files[0]);
    };

    // Function to handle training process
    const handleTrain = async () => {
        if (!file) {
        alert('Please upload a PDF file before training.');
        return;
        }

        const formData = new FormData();
        formData.append('document', file);

        try {
        const response = await axios.post('http://localhost:5000/train', formData, {
            headers: {
            'Content-Type': 'multipart/form-data',
            },
        });

        if (response.data.success) {
            alert('Training completed successfully!');
            setIsTrained(true);
        } else {
            alert('Training failed. Please try again.');
        }
        } catch (error) {
        console.error('Error during training:', error);
        console.log('Error is',error)
        alert('An error occurred during training. Please try again.');
        }
    };

    // Function to handle question submission
    const handleAsk = async () => {
        if (!isTrained) {
        alert('Please train the model before asking questions.');
        return;
        }

        try {
        const response = await axios.post('http://localhost:5000/ask', { question });

        setAnswer(response.data.answer || 'No answer received.');
        } catch (error) {
        console.error('Error during question submission:', error);
        alert('An error occurred while asking the question. Please try again.');
        }
    };

    return (
        <div style={{ textAlign: 'center', marginTop: '50px' }}>
        <h1>Sample Rag Website</h1>
        <p>Please upload the document in PDF format</p>
        <input type="file" accept="application/pdf" onChange={handleFileUpload} />
        <br />
        <button onClick={handleTrain} style={{ marginTop: '20px' }}>Train</button>
        {isTrained && (
            <div style={{ marginTop: '30px' }}>
            <input
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask your question here"
                style={{ width: '300px', padding: '10px' }}
            />
            <br />
            <button onClick={handleAsk} style={{ marginTop: '20px' }}>Ask</button>
            {answer && (
                <div style={{ marginTop: '20px', border: '1px solid #ccc', padding: '10px' }}>
                <strong>Answer:</strong>
                <p>{answer}</p>
                </div>
            )}
            </div>
        )}
        </div>
    );
};

export default RagWebsite;
