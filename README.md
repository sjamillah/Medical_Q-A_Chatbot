# Medical Question Answering System

A medical question-answering system built with TensorFlow and DistilBERT that provides accurate responses to healthcare queries using extractive question answering.

## Overview

This system addresses the need for reliable medical information access by providing immediate responses to health-related questions. It uses an extractive approach to ensure answers are grounded in verified medical content, preventing generation of false medical information.

**Key Results:**
- Token F1 score: 54.1%
- Exact Match accuracy: 3% 

## Installation

### Requirements

- Python 3.8+
- TensorFlow 2.12.0+
- 8GB RAM minimum

### Setup

```bash
git clone https://github.com/sjamillah/medical-q-a-chatbot.git
cd medical-q-a-chatbot
pip install -r requirements.txt
python main.py
```

Open your browser to `http://localhost:7860`

## Usage

### Python API

```python
from medical_qa import MedicalQASystem

qa_system = MedicalQASystem()
response = qa_system.answer_question("What are the symptoms of diabetes?")

print(f"Answer: {response['answer']}")
print(f"Confidence: {response['confidence']:.2f}")
```

### Web Interface

Start the interface with `python app.py` and navigate to the provided URL. Type your medical question and click "Get Answer".

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
├── interface/
├── dataset/
├── models/
├── notebooks/
├── requirements.txt
└── main.py
```

## Training

To train your own model:

```bash
python src/data/prepare_data.py
python src/models/train_model.py --config config/training_config.yaml
python src/models/evaluate_model.py --model_path models/your_model
```

## Testing

```bash
python -m pytest tests/
```

## Configuration

Key configuration options in `config.py`:

```python
MODEL_CONFIG = {
    "model_name": "distilbert-base-uncased-distilled-squad",
    "max_length": 512,
    "batch_size": 8,
    "confidence_threshold": 0.7
}
```

## Safety and Disclaimers

**Important:** This system provides educational information only and should never replace professional medical advice. For medical emergencies, contact emergency services immediately.

Safety features:
- Extractive approach prevents false information generation
- Confidence scoring for reliability assessment
- Integrated medical disclaimers
- Out-of-domain detection

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

Areas for contribution:
- Expanding medical categories
- Improving preprocessing
- Adding multilingual support
- Performance optimizations

## Citation

```
@article{ssozi2025medical,
  title={Medical Question Answering System using DistilBERT},
  author={Ssozi, Jamillah},
  year={2025}
}
```

## Contact

For questions or issues, please open a GitHub issue or contact [j.ssozi@alustudent.com]

---

**Medical Disclaimer:** This software is for educational purposes only. It does not provide medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions.
