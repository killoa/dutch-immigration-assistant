import os
import tempfile
import json
import logging
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from langchain_community.vectorstores import FAISS
# Fixed import - use the new package
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
## Removed TTS and STT dependencies
import warnings
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import lru_cache

# Suppress deprecation warnings to reduce noise
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables and caching
temp_files = []
vectorstore_cache = {}
embedding_model = None
executor = ThreadPoolExecutor(max_workers=4)

# Initialize embedding model once at startup
@lru_cache(maxsize=1)
def get_embedding_model():
    """Get or create the embedding model (cached)"""
    global embedding_model
    if embedding_model is None:
        logger.info("Initializing embedding model...")
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("Embedding model initialized")
    return embedding_model

def cleanup_temp_files():
    """Clean up temporary files"""
    global temp_files
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Removed temp file: {file_path}")
        except Exception as e:
            logger.warning(f"Could not remove temp file {file_path}: {e}")
    temp_files = []

def process_single_pdf(pdf_file, temp_dir):
    """Process a single PDF file efficiently"""
    try:
        temp_path = os.path.join(temp_dir, f"temp_{pdf_file.filename}")
        pdf_file.save(temp_path)
        
        loader = PyPDFLoader(temp_path)
        pages = loader.load()
        
        # Clean up immediately after loading
        try:
            os.remove(temp_path)
        except:
            pass
            
        logger.info(f"Processed {pdf_file.filename}: {len(pages)} pages")
        return pages
    except Exception as e:
        logger.error(f"Error processing {pdf_file.filename}: {e}")
        return []

def create_vectorstore_optimized(all_pages):
    """Create vectorstore with optimizations"""
    try:
        # Use smaller chunks for faster processing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Reduced chunk size for faster processing
            chunk_overlap=50,  # Reduced overlap
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        docs = text_splitter.split_documents(all_pages)
        
        # Limit number of chunks to prevent slowdown
        if len(docs) > 100:  # Limit chunks for faster processing
            docs = docs[:100]
            logger.info(f"Limited to first 100 chunks for performance")
        
        embedding_function = get_embedding_model()
        vectorstore = FAISS.from_documents(docs, embedding_function)
        logger.info(f"Created vectorstore with {len(docs)} document chunks")
        return vectorstore
    except Exception as e:
        logger.error(f"Error creating vectorstore: {e}")
        return None





# --- Main Chat Endpoint (Optimized) ---
@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint with Dutch immigration expertise"""
    try:
        # Parse payload
        logger.info(f"Request Content-Type: {request.content_type}")
        logger.info(f"Request data: {request.get_data()}")
        
        if request.is_json:
            logger.info("Processing JSON request")
            payload = request.get_json()
        else:
            logger.info("Processing form request")
            payload = request.form.get('payload')
            if not payload:
                logger.error("No payload provided in form data")
                return jsonify({'error': 'No payload provided'}), 400
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON payload: {e}")
                return jsonify({'error': 'Invalid JSON payload'}), 400
        
        logger.info(f"Processed payload: {payload}")

            
        message = payload.get('message', '').strip()
        if not message:
            return jsonify({'error': 'No message provided'}), 400
            
        history = payload.get('history', [])
        temperature = float(payload.get('temperature', 0.7))
        
        # Validate temperature
        temperature = max(0.0, min(1.0, temperature))
        
        logger.info(f"Processing immigration query: {message[:100]}...")
        
        # Handle PDFs with optimization
        pdf_files = request.files.getlist('pdfs')
        vectorstore = None
        
        # Use default vectorstore if no PDFs uploaded
        if not pdf_files or not any(f.filename for f in pdf_files):
            vectorstore = default_vectorstore
            if vectorstore:
                logger.info("Using pre-loaded immigration documents")
        else:
            start_time = logging.time.time() if hasattr(logging, 'time') else 0
            logger.info(f"Processing {len(pdf_files)} immigration documents...")
            
            # Limit number of PDFs for performance
            if len(pdf_files) > 6:
                pdf_files = pdf_files[:6]
                logger.warning("Limited to first 6 documents for performance")
            
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                all_pages = []
                
                # Process PDFs in parallel for smaller files
                if len(pdf_files) == 1:
                    # Single file - process directly
                    for pdf in pdf_files:
                        if pdf.filename:
                            pages = process_single_pdf(pdf, temp_dir)
                            all_pages.extend(pages)
                else:
                    # Multiple files - could use threading but keep simple for now
                    for pdf in pdf_files:
                        if pdf.filename:
                            pages = process_single_pdf(pdf, temp_dir)
                            all_pages.extend(pages)
                            
                            # Limit total pages for performance
                            if len(all_pages) > 50:  # Limit total pages
                                all_pages = all_pages[:50]
                                logger.warning("Limited to first 50 pages for performance")
                                break
                
                if all_pages:
                    logger.info(f"Total immigration document pages loaded: {len(all_pages)}")
                    vectorstore = create_vectorstore_optimized(all_pages)
                    
                    if hasattr(logging, 'time'):
                        processing_time = logging.time.time() - start_time
                        logger.info(f"Document processing completed in {processing_time:.2f} seconds")
        
        # Prepare conversation context with Dutch immigration focus
        conversation_messages = []
        for m in history[-5:]:
            if m.get('role') in ['user', 'assistant'] and m.get('content'):
                conversation_messages.append({
                    "role": m['role'], 
                    "content": m['content'][:500]
                })
        
        # Get API key from environment variable
        GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
        if not GROQ_API_KEY:
            print("   Please set it before starting the server.")
            GROQ_API_KEY = None

        model_name = os.environ.get('GROQ_MODEL', 'llama-3.3-70b-versatile')
        
        # Initialize LLM with Dutch immigration expertise
        try:
            groq_llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name=model_name, 
                temperature=temperature,
                max_tokens=1024,
                timeout=20
            )
            logger.info(f"Initialized Groq LLM for immigration consultation: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing Groq LLM: {e}")
            return jsonify({'error': 'Failed to initialize AI model'}), 500
        
        # Generate response with Dutch immigration focus
        if not vectorstore:
            try:
                # Add system message for immigration expertise
                conversation_messages.insert(0, {
                    "role": "system",
                    "content": "You are a Dutch immigration expert. Provide accurate information about immigration processes, visa requirements, residence permits, and integration in the Netherlands. Base your answers on official Dutch immigration policies and regulations."
                })
                conversation_messages.append({"role": "user", "content": message})
                response = groq_llm.invoke(conversation_messages)
                answer = response.content if hasattr(response, 'content') else str(response)
                sources = []
                logger.info("Generated immigration advice without documents")
            except Exception as e:
                logger.error(f"Error in immigration consultation: {e}")
                return jsonify({'error': 'Failed to generate response'}), 500
        else:
            try:
                # Simplified context window with immigration focus
                context_window = "\n".join([
                    f"{m['role']}: {m['content'][:200]}"
                    for m in conversation_messages[-3:]
                ])
                
                # Immigration-focused prompt template
                prompt_template = PromptTemplate(
                    template=(
                        "You are a Dutch immigration expert. Use the provided documents and conversation history to answer questions about immigration to the Netherlands.\n\n"
                        "Focus on providing accurate information about:\n"
                        "- Visa requirements and procedures\n"
                        "- Residence permits and their conditions\n"
                        "- Integration requirements and processes\n"
                        "- Official documentation needs\n\n"
                        f"Recent conversation:\n{context_window}\n\n"
                        "Context: {context}\n\n"
                        "Question: {question}\n\n"
                        "Answer:"
                    ),
                    input_variables=["context", "question"]
                )
                
                qa_chain = RetrievalQA.from_chain_type(
                    llm=groq_llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": prompt_template}
                )
                
                response = qa_chain.invoke({"query": message})
                answer = response.get('result', 'No answer generated')
                
                # Extract sources
                source_docs = response.get('source_documents', [])
                sources = []
                for i, doc in enumerate(source_docs[:2]):
                    content = doc.page_content.strip()
                    if len(content) > 100:
                        content = content[:100] + "..."
                    sources.append(f"Source {i+1}: {content}")
                
                logger.info("Generated immigration advice with document context")
                    
            except Exception as e:
                logger.error(f"Error in immigration document analysis: {e}")
                return jsonify({'error': 'Failed to process immigration documents'}), 500
        
        # Clean up old temp files periodically
        if len(temp_files) > 15:
            cleanup_temp_files()
        return jsonify({
            'response': answer,
            'sources': sources
        })
        
    except Exception as e:
        logger.error(f"Unexpected error in immigration consultation endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500



# --- Health check endpoint ---
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ready',
        'expertise': 'Dutch Immigration Specialist',
        'service': 'PDF-based Immigration Consultation',
        'capabilities': [
            'Dutch Immigration Law Analysis',
            'Visa Requirements Assessment',
            'Residence Permit Guidance',
            'Integration Process Information'
        ],
        'version': '1.0.0',
        'temp_files_count': len(temp_files),
        'embedding_model_loaded': embedding_model is not None
    })

# --- Progress endpoint for file upload ---
@app.route('/upload-progress', methods=['GET'])
def upload_progress():
    """Get upload progress"""
    return jsonify({
        'status': 'processing',
        'message': 'File upload in progress...'
    })

# --- Cleanup on shutdown ---
@app.teardown_appcontext
def cleanup(error):
    """Clean up resources"""
    if len(temp_files) > 30:  # Reduced threshold
        cleanup_temp_files()

# Add after imports
def load_default_pdfs():
    """Load PDFs from the pdfs directory"""
    pdf_dir = os.path.join(os.path.dirname(__file__), 'pdfs')
    if not os.path.exists(pdf_dir):
        logger.warning("PDFs directory not found")
        return None
    
    logger.info("Loading default immigration documents from pdfs directory...")
    all_pages = []
    
    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.endswith('.pdf'):
            file_path = os.path.join(pdf_dir, pdf_file)
            try:
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                all_pages.extend(pages)
                logger.info(f"Loaded {pdf_file}: {len(pages)} pages")
            except Exception as e:
                logger.error(f"Error loading {pdf_file}: {e}")
    
    if all_pages:
        logger.info(f"Total pages loaded from default documents: {len(all_pages)}")
        return create_vectorstore_optimized(all_pages)
    return None

# Add global variable after other globals
default_vectorstore = None

# Modify if __name__ == '__main__': section
if __name__ == '__main__':
    # Initialize embedding model at startup
    logger.info("Pre-loading embedding model...")
    get_embedding_model()
    
    # Load default PDFs
    logger.info("Loading default immigration documents...")
    default_vectorstore = load_default_pdfs()
    
    # Verify required environment variables
    if not os.environ.get('GROQ_API_KEY'):
        print("WARNING: GROQ_API_KEY not set in environment variables")
        print("Please set GROQ_API_KEY=your_api_key_here")
    
    print("Starting Dutch Immigration Consultation Service...")
    print(f"Server will run on http://localhost:5000")
    print(f"Health check available at http://localhost:5000/health")
    if default_vectorstore:
        print("Immigration documents loaded and ready for consultation")
    else:
        print("No immigration documents found in pdfs directory")
    
    # Run with optimized settings
    app.run(
        host='0.0.0.0', 
        port=5000, 
        debug=False,
        threaded=True,
        use_reloader=False
    )
