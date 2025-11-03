import streamlit as st
from orchestrator import orchestrate

# Set page config
st.set_page_config(
    page_title="Power System Analysis Chatbot",
    page_icon="⚡",
    layout="centered"
)

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

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask your question here..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = orchestrate(prompt)
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
    
    Simply type your question in the chat input and get instant responses!
    """)
    
    # Add citation
    st.markdown("---")
    st.markdown("Made with ❤️ by Power Systems Team")