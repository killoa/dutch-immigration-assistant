import streamlit as st
import requests
import json

# Set page config
st.set_page_config(
    page_title="Dutch Immigration Assistant",
    page_icon="🇳🇱",
    layout="centered"
)

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Backend URL
BACKEND_URL = "http://localhost:5000"

def check_backend_health():
    try:
        response = requests.get(f"{BACKEND_URL}/health")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def send_message(message, files=None):
    try:
        # Prepare the payload
        payload = {
            "message": message,
            "history": st.session_state.messages,
            "temperature": 0.7
        }
        
        # Prepare files if any
        files_data = None
        if files:
            files_data = [("pdfs", (file.name, file.getvalue(), "application/pdf")) for file in files]
        
        # Send request
        response = requests.post(
            f"{BACKEND_URL}/chat",
            data={"payload": json.dumps(payload)},
            files=files_data
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.json().get('error', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error communicating with backend: {str(e)}")
        return None

# Page header
st.title("🇳🇱 Dutch Immigration Assistant")

# Check backend health
health = check_backend_health()
if health:
    st.success("✅ Connected to Immigration Service")
    st.write(f"**Expertise:** {health.get('expertise')}")
    
    capabilities = health.get('capabilities', [])
    if capabilities:
        st.write("**Capabilities:**")
        for cap in capabilities:
            st.write(f"- {cap}")
else:
    st.error("❌ Cannot connect to backend service")
    st.stop()

# File uploader
st.write("\n")
st.write("📄 Upload immigration documents (optional)")
uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload relevant immigration documents for more specific answers"
)

# Chat interface
st.write("\n")
st.write("💬 Chat with Immigration Assistant")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("sources"):
            st.write("\n**Sources:**")
            for source in message["sources"]:
                st.write(f"- {source}")

# Chat input
if prompt := st.chat_input("Ask about immigration to the Netherlands..."):
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = send_message(prompt, uploaded_files)
            if response:
                st.write(response["response"])
                if response.get("sources"):
                    st.write("\n**Sources:**")
                    for source in response["sources"]:
                        st.write(f"- {source}")
                
                # Add assistant response to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["response"],
                    "sources": response.get("sources", [])
                })

# Sidebar with additional information
with st.sidebar:
    st.title("ℹ️ About")
    st.write("""
    This assistant helps with questions about immigration to the Netherlands.
    It can provide information about:
    - Visa requirements
    - Residence permits
    - Integration process
    - Documentation needs
    """)
    
    st.write("\n")
    st.write("**How to use:**")
    st.write("""
    1. Optionally upload relevant PDF documents
    2. Ask your questions about Dutch immigration
    3. Get detailed answers with source references
    """)