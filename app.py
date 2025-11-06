# Multi-Modal Assistant Streamlit Application
# This app provides both image analysis and text Q&A capabilities

import streamlit as st
from PIL import Image
from backend.model import get_model_response
from backend.utils import init_session_state

# Configure Streamlit page settings
st.set_page_config(page_title="🧠 Multi-Modal Assistant", layout="centered")

# Initialize session state for chat history and user data
init_session_state()
st.title("🧠 Multi-Modal Assistant")

# 🔄 Mode Selection Sidebar
st.sidebar.markdown("### Select Mode")
mode = st.sidebar.radio(
    "Select Mode",  # Empty label since we have the header
    ["Image + Text", "Text Only"],
    index=0,
    label_visibility="collapsed"
)
st.sidebar.markdown("---")


if mode == "Image + Text":
    # 📷 Image Upload and Display Section
    st.subheader("📷 Image Analysis")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        # Convert image to RGB format for model compatibility
        image = Image.open(uploaded_image).convert("RGB")
        st.session_state.image = image
        st.image(image, caption="Uploaded Image", width='stretch')
    else:
        # Clear image from session if none uploaded
        st.session_state.image = None
        st.info("ℹ️ Please upload an image to ask questions about it.")
        
elif mode == "Text Only":
    # 📝 Text Input Section for Document Q&A
    st.subheader("📝 Enter Your Text")
    text_context = st.text_area("Paste your text here (e.g., article, paragraph, document)", 
                              height=200, 
                              value=st.session_state.text_context)
    # Store text context in session state for persistence
    st.session_state.text_context = text_context
    
    if not text_context.strip():
        st.info("ℹ️ Please enter some text to ask questions about.")

# 💬 Chat form
st.subheader("💬 Ask a Question")
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Your question:")
    submitted = st.form_submit_button("Ask")

# 🧠 Process User Question and Generate Response
if submitted and user_input.strip():
    # Add user question to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.spinner("🧠 Thinking..."):
        if mode == "Image + Text":
            # Handle image-based questions
            if st.session_state.image is None:
                response = "❗ Please upload an image first."
            else:
                response = get_model_response(image=st.session_state.image, question=user_input)
        else:  # Text Only mode
            # Handle text-based questions using document context
            if not st.session_state.text_context.strip():
                response = "❗ Please enter some text first."
            else:
                response = get_model_response(
                    image=None,  
                    question=user_input,
                    context=st.session_state.text_context
                )
    
    # Add assistant response to chat history and refresh UI
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# 💬 Display Chat History
st.subheader("💬 Chat History")
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**❓ You:** {msg['content']}")
    else:
        st.markdown(f"**🧠 Assistant:** {msg['content']}")
        st.markdown("---")  # Visual separator between conversations
