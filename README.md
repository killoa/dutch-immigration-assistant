# 🇳🇱 Dutch Immigration Assistant

An AI-powered chatbot that provides expert guidance on Dutch immigration processes, visa requirements, and residence permits using official documentation.

## Features

- 🤖 **AI-Powered Consultation**: Uses Groq's LLaMA model for intelligent responses
- 📄 **PDF Document Analysis**: Upload and analyze immigration documents
- 🇳🇱 **Dutch Immigration Expertise**: Specialized knowledge of Dutch immigration law
- 💬 **Interactive Chat Interface**: User-friendly Streamlit frontend
- 🔍 **Document Search**: Vector-based search through immigration documents

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Flask
- **AI Model**: Groq LLaMA 3.3 70B
- **Embeddings**: HuggingFace Sentence Transformers
- **Vector Store**: FAISS
- **Document Processing**: LangChain + PyPDF

## Quick Start

### Prerequisites

- Python 3.8+
- Groq API Key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dutch-immigration-assistant.git
cd dutch-immigration-assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the backend:
```bash
python backend.py
```

4. Start the frontend:
```bash
streamlit run frontend.py
```

5. Open your browser to `http://localhost:8501`

## Deployment

### Streamlit Community Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy!

### Railway/Render

1. Connect your GitHub repository
2. Set environment variables
3. Deploy backend and frontend separately

## Usage

1. **Ask Questions**: Type immigration-related questions
2. **Upload Documents**: Add PDF documents for analysis
3. **Get Expert Advice**: Receive AI-powered guidance based on official documentation

## Sample Questions

- "What documents do I need for a Dutch work visa?"
- "How long does the residence permit application take?"
- "What are the integration requirements for permanent residency?"

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Disclaimer

This tool provides general information only. Always consult official Dutch immigration authorities for legal advice.