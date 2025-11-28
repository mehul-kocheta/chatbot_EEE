import streamlit as st
from orchestrator import orchestrate
import base64
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="Power System Analysis Chatbot",
    page_icon="⚡",
    layout="centered"
)

# Custom CSS to make file uploader more compact
st.markdown("""
<style>
    /* Make file uploader more compact */
    div[data-testid="stFileUploader"] {
        width: 100%;
        margin-bottom: 0.5rem;
    }
    
    /* Reduce padding in file uploader */
    div[data-testid="stFileUploader"] > div {
        padding: 0.25rem;
    }
    
    /* Make the drag and drop area more compact */
    div[data-testid="stFileUploader"] > div > div {
        min-height: 3rem;
        padding: 0.5rem;
    }
    
    /* Style the file uploader text */
    div[data-testid="stFileUploader"] label {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Add title and description
st.title("⚡ Power System Analysis Chatbot")
st.markdown("""
This chatbot can help you with:
- Power Flow Analysis
- Bus Voltage Calculations
- System Loss Analysis
- General Power System Questions
""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize uploaded image in session state
if "uploaded_image_bytes" not in st.session_state:
    st.session_state.uploaded_image_bytes = None

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], dict):
            # Handle messages with images
            if "text" in message["content"]:
                st.markdown(message["content"]["text"])
            if "image" in message["content"]:
                st.image(message["content"]["image"], caption="Uploaded Image", use_container_width=True)
        else:
            st.markdown(message["content"])

# File uploader section - compact and above chat input
uploaded_file = st.file_uploader(
    "📎 Attach image (optional)",
    type=["png", "jpg", "jpeg", "gif", "webp"],
    help="Click to upload an image or drag and drop",
    key="image_uploader"
)

# Show image preview if uploaded
if uploaded_file is not None:
    # Read and store image bytes
    image_bytes = uploaded_file.read()
    st.session_state.uploaded_image_bytes = image_bytes
    # Show compact preview
    preview_image = Image.open(io.BytesIO(image_bytes))
    st.image(preview_image, caption="✅ Image ready to send", width=200)
elif st.session_state.uploaded_image_bytes is not None:
    # Show preview if image is still in session state (before sending)
    preview_image = Image.open(io.BytesIO(st.session_state.uploaded_image_bytes))
    st.image(preview_image, caption="✅ Image ready to send", width=200)

# Chat input
prompt = st.chat_input("Ask your question here...")

# Process chat input
if prompt:
    # Convert uploaded image to base64 if available
    image_base64 = None
    image_display = None
    
    if st.session_state.uploaded_image_bytes is not None:
        # Use stored image bytes
        image_bytes = st.session_state.uploaded_image_bytes
        image_display = Image.open(io.BytesIO(image_bytes))
        
        # Convert to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Clear uploaded image after use
        st.session_state.uploaded_image_bytes = None
    
    # Prepare message content for display
    message_content = prompt
    if image_display:
        message_content = {"text": prompt, "image": image_display}
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
        if image_display:
            st.image(image_display, caption="Uploaded Image", use_container_width=True)
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": message_content})
    
    # Prepare conversation history for orchestrate
    conversation_history = []
    for msg in st.session_state.messages[:-1]:  # Exclude the current message
        if isinstance(msg["content"], dict):
            # Convert image message to text-only for history
            conv_msg = {
                "role": msg["role"],
                "content": msg["content"].get("text", "")
            }
        else:
            conv_msg = {
                "role": msg["role"],
                "content": msg["content"]
            }
        conversation_history.append(conv_msg)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = orchestrate(prompt, image_base64=image_base64, conversation_history=conversation_history)
            st.markdown(response)
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add sidebar with information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This chatbot uses advanced AI to help with power system analysis tasks. It can:
    
    1. Solve power flow problems
    2. Calculate system losses
    3. Answer general power system questions
    4. Provide web search results for broader topics
    5. Analyze images related to power systems
    
    Simply type your question in the chat input and attach images using the 📎 button to get instant responses!
    """)
    
    # Add citation
    st.markdown("---")
    st.markdown("Made with ❤️ by Power Systems Team")