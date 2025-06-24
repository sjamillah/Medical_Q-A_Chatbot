# Medical Q&A Assistant

An AI-powered medical question-answering assistant built with TensorFlow and DistilBERT. This application provides educational medical information through an extractive question-answering approach.

## Features

- **Advanced NLP Model**: Uses DistilBERT fine-tuned for question answering
- **Medical Context Awareness**: Specialized preprocessing for medical terminology
- **Confidence Scoring**: Provides confidence levels for each answer
- **User-Friendly Interface**: Clean, professional Gradio interface
- **Comprehensive Disclaimer**: Clear medical advisory warnings

## Model Details

- **Framework**: TensorFlow/Keras
- **Base Model**: DistilBERT (distilbert-base-uncased-distilled-squad)
- **Task**: Extractive Question Answering
- **Training Data**: Medical Q&A dataset (MedQuad)
- **Max Sequence Length**: 512 tokens

## Usage

Simply type your medical question in the text box and click "Get Answer". The system will:

1. Process your question using medical-aware text cleaning
2. Find relevant context or generate appropriate medical context
3. Extract the most relevant answer using the trained model
4. Provide a confidence score for the answer

## Example Questions

- "What are the symptoms of diabetes?"
- "How can I treat a headache naturally?"
- "What causes high blood pressure?"
- "What should I do if I have a fever?"

## Important Disclaimer

⚠️ **This application is for educational and informational purposes only.** 

- It should NOT replace professional medical advice, diagnosis, or treatment
- Always consult qualified healthcare providers for medical concerns
- In case of emergencies, contact emergency services immediately
- Do not use this tool for self-diagnosis or treatment decisions

## Technical Implementation

The application implements:

- **Text Preprocessing**: Medical stopword analysis and context-aware cleaning
- **Model Architecture**: TensorFlow implementation of DistilBERT for QA
- **Answer Extraction**: Span-based answer extraction with confidence scoring
- **Context Generation**: Fallback medical context generation for better answers

## Limitations

- Answers are extracted from provided contexts, not generated
- Performance depends on the relevance of available medical contexts
- Should not be used for emergency medical situations
- Accuracy may vary depending on question complexity and medical domain

## Development

Built with:
- TensorFlow 2.12+
- Transformers (Hugging Face)
- Gradio 4.0+
- NLTK for text processing

For local development, install dependencies:
```bash
pip install -r requirements.txt
python app.py
```

## License

MIT License - See LICENSE file for details.

## Contact

For issues or questions about this application, please open an issue in the repository.
