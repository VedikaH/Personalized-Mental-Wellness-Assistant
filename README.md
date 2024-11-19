

# Personalized-Mental-Wellness-Assistant


This repository contains the source code for the **Personalized-Mental-Wellness-Assistant**, a conversational AI designed to provide empathetic and supportive responses to users seeking mental health guidance. The chatbot leverages the **LLaMA-2** model fine-tuned for mental health conversations and is integrated with a frontend using **Streamlit** for user interaction. 


## Overview

The **Personalized-Mental-Wellness-Assistant** is an AI-powered tool that provides users with thoughtful, compassionate, and non-judgmental responses. It is designed to simulate therapy-like conversations and offer suggestions and coping mechanisms for users struggling with feelings like loneliness, stress, or anxiety.

This chatbot:
- Understands user input and provides relevant advice and coping strategies.
- Maintains context within a conversation, allowing for a more personalized experience.
- Is not a replacement for professional therapy but aims to offer immediate emotional support.

## Features

- **Empathetic Responses**: Provides kind and understanding replies to mental health queries.
- **Context Memory**: Retains conversation history in a single session to offer context-aware responses.
- **Frontend UI**: Built using **Streamlit** to provide a user-friendly chat interface.
- **Fine-Tuned Model**: Based on LLaMA-2 and fine-tuned for mental health conversations.
  
## Model Details

- **Base Model**: [LLaMA-2-7b](https://huggingface.co/meta-llama/Llama-2-7b) by Meta AI.
- **Fine-Tuned Model**: `Vedika8/Therapy_chatbot` on Hugging Face.
- **Training Approach**: Fine-tuned using QLoRA (Quantized LoRA) for resource-efficient training.
- **Language**: English.

## Dataset

The model is fine-tuned on a curated dataset of mental health-related conversations. The dataset includes examples of:
- Common mental health topics such as anxiety, loneliness, stress, etc.
- Coping strategies and suggestions to address various emotional struggles.
  
This dataset was prepared to focus on providing supportive and non-judgmental responses.

## Installation

### Prerequisites

- Python 3.8+
- PyTorch with CUDA (optional for GPU support)
- Hugging Face `transformers` library
- `streamlit` for the frontend UI

### Clone the Repository

```bash
git clone https://github.com/your-username/Mental-Health-Therapy-Chatbot.git
cd Mental-Health-Therapy-Chatbot
```

### Hugging Face Authentication

To use the fine-tuned model, you'll need to authenticate with Hugging Face. You can do this by running:

```bash
huggingface-cli login
```

Then enter your Hugging Face token, which you can obtain from your [Hugging Face profile](https://huggingface.co/settings/tokens).

## Usage

### Running Locally

To run the chatbot locally using Streamlit, simply execute the following command:

```bash
streamlit run app.py
```

This will launch the chatbot in your browser.


## Frontend (Streamlit)

The chatbot uses **Streamlit** for its user interface. Users can type their messages into the input box and receive responses from the chatbot. To reset the conversation, there’s a **New Chat** button that clears the session state.

## Context Memory

The chatbot maintains context during a conversation by storing the user's previous inputs and responses in the session state. This allows the bot to generate more personalized and context-aware responses.

### How It Works

- Each user input and the corresponding chatbot response are stored in `st.session_state`.
- When the user types a new message, the bot considers the entire conversation history before generating a new response.

If you'd like to clear the chat history, simply click the **"New Chat"** button.

## Limitations

- **Non-Clinical Tool**: This chatbot is not intended to replace professional mental health counseling. It is designed for supportive conversations and general advice, but users facing severe mental health challenges should consult licensed professionals.
- **Model Limitations**: The model may sometimes produce responses that are too general or fail to fully understand the nuances of a specific emotional situation.



