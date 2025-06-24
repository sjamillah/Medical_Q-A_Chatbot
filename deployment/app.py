import gradio as gr
import tensorflow as tf
from transformers import AutoTokenizer, TFDistilBertForQuestionAnswering
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    print("NLTK download failed, using fallback stopwords")

class MedicalQAChatbot:
    def __init__(self):
        """Initialize the medical QA chatbot"""
        self.max_length = 512
        self.model = None
        self.tokenizer = None
        self.session_start = datetime.now()
        self.question_count = 0
        
        # Medical stopwords to keep (based on your analysis)
        self.medical_keep = {
            'no', 'not', 'never', 'none', 'neither', 'nor',
            'can', 'cannot', 'could', 'should', 'would', 'may', 'might', 'must',
            'what', 'when', 'where', 'why', 'how', 'which',
            'if', 'while', 'before', 'after', 'during', 'with', 'without'
        }
        
        self.load_model()
    
    def load_model(self):
        """Load the model and tokenizer"""
        try:
            print("Loading model and tokenizer...")
            
            # Try to load from local files first, then fallback to pre-trained
            model_paths = [
                "./medical_qa_model_transformers",
                "./",
                "distilbert-base-uncased-distilled-squad"
            ]
            
            for model_path in model_paths:
                try:
                    if os.path.exists(model_path) or model_path.startswith("distilbert"):
                        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                        self.model = TFDistilBertForQuestionAnswering.from_pretrained(model_path)
                        print(f"Model loaded successfully from {model_path}")
                        break
                except Exception as e:
                    print(f"Failed to load from {model_path}: {e}")
                    continue
            
            if self.model is None:
                raise Exception("Could not load any model")
            
            # Set up TensorFlow optimizations
            tf.random.set_seed(42)
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    
    def clean_text(self, text, remove_stopwords=False):
        """Clean text with medical context awareness"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep medical-relevant punctuation
        text = re.sub(r'[^\w\s\?\.\!,\-\']', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        if remove_stopwords:
            try:
                stop_words = set(stopwords.words('english'))
                # Remove medical-important stopwords from removal list
                stop_words = stop_words - self.medical_keep
                words = text.split()
                words = [word for word in words if word not in stop_words]
                return ' '.join(words)
            except:
                # Fallback if stopwords not available
                return text
        
        return text
    
    def get_medical_context_fallback(self, question):
        """Generate relevant medical context based on question keywords"""
        medical_contexts = {
            "diabetes": "Diabetes is a chronic condition affecting blood sugar levels. Type 1 diabetes occurs when the pancreas produces little or no insulin. Type 2 diabetes occurs when the body becomes resistant to insulin or doesn't produce enough insulin to maintain normal glucose levels.",
            
            "headache": "Headaches can be primary (like migraines, tension headaches, or cluster headaches) or secondary (caused by another condition). Common treatments include rest, hydration, over-the-counter pain relievers, stress management, and identifying triggers.",
            
            "blood pressure": "High blood pressure (hypertension) occurs when blood force against artery walls is consistently too high. Normal blood pressure is below 120/80 mmHg. High blood pressure can lead to heart disease, stroke, and kidney problems if left untreated.",
            
            "fever": "Fever is a temporary increase in body temperature, often due to illness or infection. Normal body temperature is around 98.6°F (37°C). Fever is usually a sign that the body is fighting an infection.",
            
            "flu": "Influenza (flu) is a respiratory illness caused by influenza viruses. Symptoms include fever, cough, body aches, fatigue, and sometimes vomiting and diarrhea. The flu vaccine can help prevent infection.",
            
            "heart": "The heart is a muscular organ that pumps blood throughout the body. Heart disease includes conditions like coronary artery disease, heart attack, heart failure, and arrhythmias. Regular exercise and a healthy diet support heart health.",
            
            "infection": "Infections are caused by microorganisms like bacteria, viruses, fungi, or parasites. The body's immune system fights infections, and treatments may include antibiotics for bacterial infections, antivirals for viral infections, or antifungals.",
            
            "pain": "Pain is the body's way of signaling injury or illness. It can be acute (short-term) or chronic (long-term). Pain management includes medications, physical therapy, relaxation techniques, and addressing underlying causes.",
            
            "medication": "Medications are substances used to treat, prevent, or diagnose diseases. They work by interacting with the body's biological processes. It's important to take medications as prescribed and be aware of potential side effects.",
            
            "vaccine": "Vaccines help the immune system recognize and fight specific diseases. They contain weakened or inactive parts of organisms that cause disease. Vaccines are one of the most effective ways to prevent infectious diseases."
        }
        
        question_lower = question.lower()
        
        # Find relevant context based on keywords
        for keyword, context in medical_contexts.items():
            if keyword in question_lower:
                return context
        
        # Default context for general medical queries
        return f"Medical information and healthcare guidance. This appears to be a medical query that may require professional medical consultation for accurate diagnosis and treatment recommendations."
    
    def extract_answer(self, question, context=None):
        """Extract answer using the trained model"""
        try:
            # Clean the question
            clean_question = self.clean_text(question)
            
            # Use provided context or generate one
            if context is None:
                context = self.get_medical_context_fallback(clean_question)
            
            # Tokenize input
            inputs = self.tokenizer(
                clean_question,
                context,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors="tf"
            )
            
            # Get model outputs
            outputs = self.model({
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask']
            })
            
            # Extract answer span
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            
            start_idx = tf.argmax(start_logits, axis=1)[0]
            end_idx = tf.argmax(end_logits, axis=1)[0]
            
            # Ensure valid span
            if end_idx < start_idx:
                end_idx = start_idx
            
            # Extract answer tokens
            answer_tokens = inputs["input_ids"][0][start_idx:end_idx+1]
            answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
            
            # Calculate confidence
            start_probs = tf.nn.softmax(start_logits, axis=1)[0]
            end_probs = tf.nn.softmax(end_logits, axis=1)[0]
            confidence = (start_probs[start_idx] + end_probs[end_idx]) / 2
            
            return {
                "answer": answer.strip(),
                "confidence": float(confidence),
                "start_position": int(start_idx),
                "end_position": int(end_idx)
            }
            
        except Exception as e:
            print(f"Error in answer extraction: {e}")
            return {
                "answer": "I apologize, but I encountered an error processing your question. Please try rephrasing your question.",
                "confidence": 0.0,
                "start_position": 0,
                "end_position": 0
            }
    
    def answer_question(self, question, custom_context=None):
        """Main function to answer medical questions"""
        self.question_count += 1
        
        if not question or not question.strip():
            return {
                "answer": "Please ask a medical question and I'll do my best to help you.",
                "confidence": 0.0,
                "confidence_level": "No question provided",
                "disclaimer": "This information is for educational purposes only. Always consult healthcare professionals for medical advice."
            }
        
        # Extract answer using the model
        result = self.extract_answer(question, custom_context)
        
        # Determine confidence level
        if result["confidence"] > 0.8:
            confidence_level = "High confidence"
        elif result["confidence"] > 0.5:
            confidence_level = "Moderate confidence"
        elif result["confidence"] > 0.2:
            confidence_level = "Low confidence"
        else:
            confidence_level = "Very low confidence"
        
        # Check if we have a meaningful answer
        if result["answer"] and len(result["answer"].strip()) > 2:
            return {
                "answer": result["answer"],
                "confidence": result["confidence"],
                "confidence_level": confidence_level,
                "disclaimer": "This information is for educational purposes only. Always consult healthcare professionals for medical advice."
            }
        else:
            return {
                "answer": "I couldn't find a specific answer to your question. Please try rephrasing your question or providing more context. For specific medical concerns, please consult a healthcare professional.",
                "confidence": 0.0,
                "confidence_level": "No answer found",
                "disclaimer": "This information is for educational purposes only. Always consult healthcare professionals for medical advice."
            }
    
    def get_analytics(self):
        """Get session analytics"""
        uptime = datetime.now() - self.session_start
        return {
            "Model Type": "TensorFlow DistilBERT for Question Answering",
            "Questions Answered": str(self.question_count),
            "Session Uptime": f"{uptime.seconds // 60} minutes",
            "Model Parameters": f"{self.model.count_params():,}" if self.model else "Unknown",
            "Max Sequence Length": str(self.max_length),
            "Status": "Ready"
        }

# Initialize the chatbot
print("Initializing Medical QA Chatbot...")
chatbot = MedicalQAChatbot()

def create_gradio_interface():
    """Create the main Gradio interface for Hugging Face Spaces"""
    
    # Custom CSS for professional styling
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .main-header {
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .chat-container {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .disclaimer-box {
        background: linear-gradient(45deg, #FFF3CD, #FFEAA7);
        border: 2px solid #F39C12;
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
        color: #8B4513;
        font-weight: 500;
    }
    
    .gr-button-primary {
        background: linear-gradient(45deg, #28a745, #20c997) !important;
        border: none !important;
        border-radius: 25px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .gr-button-primary:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(40, 167, 69, 0.3) !important;
    }
    
    .example-btn {
        background: linear-gradient(45deg, #17a2b8, #6f42c1) !important;
        color: white !important;
        border: none !important;
        border-radius: 15px !important;
        font-size: 12px !important;
        padding: 8px 12px !important;
        margin: 2px !important;
    }
    
    .analytics-box {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #007bff;
    }
    """
    
    def chat_function(question, chat_history, context=""):
        """Process chat messages"""
        if not question.strip():
            return "", chat_history
        
        try:
            # Get response from chatbot
            custom_context = context.strip() if context and context.strip() else None
            response_dict = chatbot.answer_question(question, custom_context)
            
            # Format response with confidence and disclaimer
            formatted_response = f"""**Answer:** {response_dict['answer']}
**Confidence:** {response_dict['confidence_level']} ({response_dict['confidence']:.1%})
---
*{response_dict['disclaimer']}*"""
            
            # Add to chat history
            chat_history.append([question, formatted_response])
            
        except Exception as e:
            error_response = f"Sorry, I encountered an error: {str(e)}"
            chat_history.append([question, error_response])
        
        return "", chat_history
    
    def get_analytics_display():
        """Format analytics for display"""
        try:
            analytics = chatbot.get_analytics()
            formatted = "## Session Analytics\n\n"
            for key, value in analytics.items():
                formatted += f"**{key}:** {value}  \n"
            return formatted
        except Exception as e:
            return f"Analytics unavailable: {str(e)}"
    
    def clear_chat():
        """Clear chat history"""
        return []
    
    # Example questions
    example_questions = [
        "What are the symptoms of diabetes?",
        "How can I treat a headache naturally?",
        "What causes high blood pressure?",
        "What are the side effects of aspirin?",
        "How can I prevent the flu?",
        "What should I do if I have a fever?",
        "What are the signs of dehydration?",
        "How much water should I drink daily?"
    ]
    
    # Create the interface
    with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Medical Q&A Assistant") as demo:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1 style="text-align: center; color: #2c3e50; margin: 0; font-size: 2.5em;">
                Medical Q&A Assistant
            </h1>
            <p style="text-align: center; color: #7f8c8d; font-size: 1.2em; margin: 10px 0 0 0;">
                AI-Powered Medical Information Assistant using TensorFlow & DistilBERT
            </p>
        </div>
        """)
        
        # Disclaimer
        gr.HTML("""
        <div class="disclaimer-box">
            <strong>Important Medical Disclaimer:</strong><br>
            This AI assistant provides general medical information for educational purposes only. 
            It should NOT replace professional medical advice, diagnosis, or treatment. 
            Always consult qualified healthcare providers for medical concerns, emergencies, or treatment decisions.
        </div>
        """)
        
        # Main interface
        with gr.Row():
            # Left column - Input and examples
            with gr.Column(scale=1):
                gr.Markdown("### Ask Your Medical Question")
                
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Example: What are the symptoms of diabetes?",
                    lines=3,
                    elem_classes="chat-input"
                )
                
                context_input = gr.Textbox(
                    label="Additional Context (Optional)",
                    placeholder="Provide any additional context or specific details...",
                    lines=2,
                    elem_classes="context-input"
                )
                
                with gr.Row():
                    submit_btn = gr.Button("Get Answer", variant="primary", scale=2)
                    clear_btn = gr.Button("Clear", scale=1)
                
                gr.Markdown("### Try These Example Questions")
                
                with gr.Row():
                    with gr.Column():
                        for i in range(0, 4):
                            btn = gr.Button(
                                example_questions[i], 
                                size="sm",
                                elem_classes="example-btn"
                            )
                            btn.click(
                                fn=lambda x=example_questions[i]: x,
                                outputs=question_input
                            )
                    
                    with gr.Column():
                        for i in range(4, 8):
                            btn = gr.Button(
                                example_questions[i], 
                                size="sm",
                                elem_classes="example-btn"
                            )
                            btn.click(
                                fn=lambda x=example_questions[i]: x,
                                outputs=question_input
                            )
                
                # Analytics section
                gr.Markdown("### Model Information")
                analytics_display = gr.Markdown(
                    value=get_analytics_display(),
                    elem_classes="analytics-box"
                )
            
            # Right column - Chat interface
            with gr.Column(scale=2):
                gr.Markdown("### Conversation")
                
                chatbot_ui = gr.Chatbot(
                    label="Medical Assistant",
                    height=600,
                    elem_classes="chat-container"
                )
        
        # Event handlers
        submit_btn.click(
            fn=chat_function,
            inputs=[question_input, chatbot_ui, context_input],
            outputs=[question_input, chatbot_ui]
        ).then(
            fn=get_analytics_display,
            outputs=analytics_display
        )
        
        question_input.submit(
            fn=chat_function,
            inputs=[question_input, chatbot_ui, context_input],
            outputs=[question_input, chatbot_ui]
        ).then(
            fn=get_analytics_display,
            outputs=analytics_display
        )
        
        clear_btn.click(
            fn=clear_chat,
            outputs=chatbot_ui
        ).then(
            fn=get_analytics_display,
            outputs=analytics_display
        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 20px; background: rgba(255,255,255,0.1); border-radius: 10px;">
            <p style="color: #6c757d; margin: 0;">
                Powered by TensorFlow, Transformers & Gradio | 
                <strong>Remember:</strong> Always consult healthcare professionals for medical advice
            </p>
        </div>
        """)
    
    return demo

if __name__ == "__main__":
    print("Starting Medical QA Chatbot Interface...")
    
    try:
        # Create and launch interface
        demo = create_gradio_interface()
        
        # Launch for Hugging Face Spaces
        demo.launch()
        
    except Exception as e:
        print(f"Error launching interface: {e}")
        # Provide a simple test
        print("Testing chatbot functionality...")
        try:
            test_response = chatbot.answer_question("What is diabetes?")
            print(f"Test successful: {test_response['answer'][:100]}...")
        except Exception as test_error:
            print(f"Test failed: {test_error}")
