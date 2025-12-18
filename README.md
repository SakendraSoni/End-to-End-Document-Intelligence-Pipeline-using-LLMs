# End-to-End-Document-Intelligence-Pipeline-using-LLMs
LLM-Powered Document Intelligence System A Python-based pipeline using pdfplumber and Hugging Face Transformers (T5, RoBERTa) to automate PDF text extraction, summarization, and interactive Q&A.
## 🌟 Features

- **PDF Text Extraction**: Extracts text content from PDF documents using `pdfplumber`
- **Document Summarization**: Generates concise summaries using T5-small transformer model
- **Intelligent Question Generation**: Automatically creates relevant questions from document passages
- **Automated Question Answering**: Answers generated questions using RoBERTa-based QA model
- **Passage Management**: Intelligently splits documents into manageable passages for better processing
- **Duplicate Question Filtering**: Ensures unique questions are processed to avoid redundancy

## 🛠️ Technologies Used

- **Python 3.12+**
- **pdfplumber**: PDF text extraction
- **Transformers (HuggingFace)**: LLM models for NLP tasks
- **NLTK**: Natural language processing and tokenization
- **PyTorch**: Deep learning framework
- **Google Colab**: Development and execution environment

## 📋 Models Used

1. **T5-Small**: Text summarization
2. **T5-Base-QG-HL (Valhalla)**: Question generation
3. **RoBERTa-Base-Squad2 (Deepset)**: Question answering

## 🚀 Getting Started

### Prerequisites

```bash
pip install pdfplumber
pip install transformers
pip install torch
pip install nltk
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/document-analysis-llm.git
cd document-analysis-llm
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```python
import nltk
nltk.download('punkt_tab')
```

## 📖 Usage

### 1. Extract Text from PDF

```python
import pdfplumber

pdf_path = 'your_document.pdf'
output_text_file = "extracted_text.txt"

with pdfplumber.open(pdf_path) as pdf:
    extracted_text = ""
    for page in pdf.pages:
        extracted_text += page.extract_text()

with open(output_text_file, "w") as text_file:
    text_file.write(extracted_text)
```

### 2. Generate Summary

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="t5-small")
summary = summarizer(document_text[:1000], max_length=150, min_length=30, do_sample=False)
print("Summary:", summary[0]['summary_text'])
```

### 3. Generate Questions

```python
qg_pipeline = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")

def generate_question_pipeline(passage, min_questions=3):
    input_text = f"generate questions: {passage}"
    results = qg_pipeline(input_text)
    questions = results[0]['generated_text'].split('<sep>')
    return [q.strip() for q in questions if q.strip()][:min_questions]
```

### 4. Answer Questions

```python
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

answer = qa_pipeline({'question': question, 'context': passage})
print(f"Q: {question}")
print(f"A: {answer['answer']}")
```

## 📁 Project Structure

```
document-analysis-llm/
│
├── End-to-End-Document-Intelligence-Pipeline-using-LLMs.ipynb    # Main notebook
├── extracted_text.txt                     # Extracted PDF content
├── requirements.txt                       # Python dependencies
├── README.md                             # Project documentation
└── sample_documents/                     # Sample PDFs for testing
```

## 🔍 How It Works

1. **Text Extraction**: The system uses `pdfplumber` to extract text from PDF documents page by page

2. **Passage Segmentation**: Extracted text is split into manageable passages (~200 words each) using NLTK tokenization

3. **Question Generation**: For each passage, the T5-based question generation model creates 3+ relevant questions

4. **Answer Extraction**: The RoBERTa QA model processes each question against its source passage to extract precise answers

5. **Duplicate Prevention**: A set-based tracking system ensures each unique question is answered only once

## 💡 Use Cases

- **Legal Document Analysis**: Extract key information from contracts and terms of service
- **Research Paper Summarization**: Quickly understand academic papers
- **Educational Content**: Generate study materials from textbooks
- **Compliance Reviews**: Analyze policy documents and regulations
- **Customer Support**: Create FAQ pairs from documentation

## 🎯 Example Output

```
Q: What does the Google Terms of Service cover?
A: laws that apply to our company

Q: What is the Privacy Policy?
A: we encourage you to read it

Q: How does Google earn money?
A: how Google's business works
```

## ⚠️ Limitations

- GPU recommended for faster processing (Colab T4 used in development)
- Large documents may require chunking for memory management
- Question quality depends on passage coherence
- Works best with well-structured English text

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- HuggingFace for providing pre-trained transformer models
- Google Colab for free GPU access
- The open-source NLP community

## 📧 Contact

Your Name - Sakendrasoni38@example.com

Project Link: [https://github.com/yourusername/document-analysis-llm](https://github.com/yourusername/End-to-End-Document-Intelligence-Pipeline-using-LLMs)

---

⭐ Star this repo if you find it helpful!
