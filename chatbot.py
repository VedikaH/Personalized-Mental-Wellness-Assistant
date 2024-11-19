
import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set up the Streamlit app
st.set_page_config(page_title="Therapy Chatbot", layout="wide")

# Custom CSS to style the chat interface
st.markdown("""
<style>
.stTextInput > div > div > input {
    border-radius: 20px;
}
.stButton > button {
    border-radius: 20px;
    float: right;
}
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
</style>
""", unsafe_allow_html=True)

# Load the model (unchanged)
@st.cache_resource
def load_model():
    model = AutoModelForCausalLM.from_pretrained("Vedika8/Therapy_chatbot", torch_dtype=torch.float16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("Vedika8/Therapy_chatbot")
    return model, tokenizer, device

model, tokenizer, device = load_model()

# Functions for prompt formatting and output cleaning (unchanged)
def format_prompt(prompt, chat_history):
    history = "".join([f"User: {entry['user']}\nAI: {entry['ai']}\n" for entry in chat_history])
    return f"[INST] <<SYS>> You are a virtual AI therapy assistant. Your role is to provide thoughtful and supportive responses. Always ensure that you complete your last sentence with a period.<</SYS>> {history}User: {prompt.strip()} [/INST]"

def clean_output(output_text, input_text):
    # Ensure special tokens are removed, but not meaningful text
    output_text = output_text.replace(input_text, "")
    output_text = output_text.replace("[INST]", "").replace("[/INST]", "").replace("(period)","").replace("(Period)","")
    output_text = output_text.replace("1)", "\n\n1)").replace("2)", "\n\n2)").replace("3)", "\n\n3)")\
        .replace("4)", "\n\n4)").replace("5)", "\n\n5)").replace("6)", "\n\n6)").replace("7)", "\n\n7)").replace("8)", "\n\n8)").replace("9)", "\n\n9)")
    return output_text.strip()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
# New Chat Button: Clears the chat history to start a new session
if st.button("New Chat"):
    st.session_state.chat_history = []

# Chat interface
st.markdown("<h1 style='text-align: center;'>Therapy Chatbot 🤗</h1>", unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(message["user"])
    with st.chat_message("assistant"):
        st.write(message["ai"])

# User input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({"user": user_input, "ai": ""})
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)

    # Generate bot response
    formatted_prompt = format_prompt(user_input, st.session_state.chat_history[:-1])
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            temperature=0.6,
            max_new_tokens=500,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    clean_response = clean_output(response, formatted_prompt)
    
    # Update the last message in chat history with bot response
    st.session_state.chat_history[-1]["ai"] = clean_response
    
    # Display bot response
    with st.chat_message("assistant"):
        st.write(clean_response)

# Clean up memory
torch.cuda.empty_cache()

import streamlit as st
from pyngrok import ngrok
import subprocess
import os

# Set your ngrok auth token
ngrok.set_auth_token("2lnntTQURjEGdZOmNXeg2WyjMoR_5zmkPGZhkXexEtUsKBaDJ")  # Replace with your actual token

# Get the dev server port (defaults to 8501)
port = 8501

# Open a ngrok tunnel to the dev server
public_url = ngrok.connect(port).public_url
print(f"Public URL: {public_url}")

# Update the environment variable for Streamlit to use
os.environ['STREAMLIT_SERVER_PORT'] = str(port)

# Run the Streamlit app
print("Running Streamlit app...")
subprocess.Popen(['streamlit', 'run', 'app.py'])

# Keep the notebook running
import time
while True:
    time.sleep(1)
