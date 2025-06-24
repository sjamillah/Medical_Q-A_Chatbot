# Medical Question Answering System

A medical question-answering system built with TensorFlow and DistilBERT that provides accurate responses to healthcare queries using extractive question answering.

## Live Demo

**Try the live application:** [Medical Q&A Assistant on Hugging Face Spaces](https://huggingface.co/spaces/Jammy142/Medical_Q-A_Chatbot)

## Overview

This system addresses the need for reliable medical information access by providing immediate responses to health-related questions. It uses an extractive approach to ensure answers are grounded in verified medical content, preventing generation of false medical information.

**Key Results:**
- Token F1 score: 54.1%
- Exact Match accuracy: 3% 
- ROUGE-L: 86.0%
- BERT Score: 95.9%

## Quick Start

### Online Demo (Recommended)
Visit our [live demo](https://huggingface.co/spaces/Jammy142/Medical_Q-A_Chatbot) - no installation required!

### Local Installation

#### Requirements
- Python 3.8+
- TensorFlow 2.12.0+
- 8GB RAM minimum

#### Setup
```bash
git clone https://github.com/sjamillah/medical_q-a_chatbot.git
cd medical_q-a_chatbot
pip install -r requirements.txt
python src/data/final_tensorflow_extractive_q&a_chatbot.py
```

## Usage

### Web Interface (Recommended)
1. **Online**: Visit the [Hugging Face Space](https://huggingface.co/spaces/Jammy142/Medical_Q-A_Chatbot)
2. **Local**: Start with `python app.py` and navigate to the provided URL
3. Type your medical question and click "Get Answer"

### Python API
```python
from medical_qa import MedicalQASystem

qa_system = MedicalQASystem()
response = qa_system.answer_question("What are the symptoms of diabetes?")
print(f"Answer: {response['answer']}")
print(f"Confidence: {response['confidence']:.2f}")
```

## Dataset

Uses the MedQuad dataset with 43,000+ medical question-answer pairs across 31 medical categories. All responses come from professional healthcare providers.

## Performance

| Metric | Score |
|--------|-------|
| Token F1 | 54.1% |
| Exact Match | 3% |
| ROUGE-L | 86.0% |
| BERT Score | 95.9% |

Performance varies by medical category:
- Diabetes Management: 84.2% F1
- Preventive Care: 81.7% F1
- Drug Interactions: 68.4% F1
- Rare Conditions: 65.1% F1

## Project Structure

```
medical-qa-system/
├── dataset/
│   └── medicalQ&A.csv
├── deployment/
│   ├── app.py                    # Gradio web interface
│   ├── requirements.txt          # Production dependencies
│   └── README.md                # Deployment documentation
├── models/
│   └── medical_qa_model_transformers/  # Trained model files
├── notebooks/                    # Jupyter notebooks for exploration
├── scripts/
│   └── final_tensorflow_extractive_q&a_chatbot.py
│   └── tensorflow_extractive_q&a_chatbot_1.py
└── requirements.txt             # Development dependencies
└── .gitattributes
└── .gitignore
└── README.md
```

## Deployment

### Hugging Face Spaces (Live Demo)
The application is deployed on Hugging Face Spaces for easy access:
- **Live URL**: [Live Demo](https://huggingface.co/spaces/Jammy142/Medical_Q-A_Chatbot)
- **Deployment files**: Located in `/deployment/` folder
- **Automatic updates**: Synced with this repository

### Local Deployment
```bash
# For training and development
python src/data/final_tensorflow_extractive_q&a_chatbot.py

# For web interface (optimized)
cd deployment/
python app.py
```

## Training

To train your own model:
```bash
python src/data/final_tensorflow_extractive_q&a_chatbot.py
```

The training script includes:
- Data preprocessing with medical stopword analysis
- Model fine-tuning on medical Q&A dataset
- Comprehensive evaluation metrics
- Model saving and checkpointing

## Testing

Tests can be added in the future for model validation and interface testing.

## Configuration

The model configuration is embedded in the training script with these key parameters:

```python
MODEL_CONFIG = {
    "model_name": "distilbert-base-uncased-distilled-squad",
    "max_length": 512,
    "batch_size": 8,
    "confidence_threshold": 0.7
}
```

## Repository Structure

- **`/src/`**: Source code and training scripts
- **`/deployment/`**: Production-ready deployment files
- **`/models/`**: Trained model artifacts
- **`/notebooks/`**: Research and development notebooks

## Safety and Disclaimers

**Important:** This system provides educational information only and should never replace professional medical advice. For medical emergencies, contact emergency services immediately.

Safety features:
- Extractive approach prevents false information generation
- Confidence scoring for reliability assessment
- Integrated medical disclaimers
- Out-of-domain detection

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test your changes locally
5. Submit a pull request

Areas for contribution:
- Expanding medical categories
- Improving preprocessing
- Adding multilingual support
- Performance optimizations
- UI/UX improvements

## Citation

```bibtex
@article{ssozi2025medical,
  title={Medical Question Answering System using DistilBERT},
  author={Ssozi, Jamillah},
  year={2025},
  url={https://github.com/sjamillah/medical_q-a_chatbot}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Author**: Jamillah Ssozi
- **Email**: [j.ssozi@alustudent.com](mailto:j.ssozi@alustudent.com)
- **GitHub**: [@sjamillah](https://github.com/sjamillah)

For questions or issues, please open a [GitHub issue](https://github.com/sjamillah/medical_q-a_chatbot/issues).

---

**Medical Disclaimer:** This software is for educational purposes only. It does not provide medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions.
