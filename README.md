
# Multi Document-URL Chatbot

## Overview

This is a multi-document and URL chatbot that supports various file formats, including PDF, DOCX, TXT, and CSV. The chatbot is designed to answer user questions based on the provided documents or URLs.

## Features

- Multiple File Formats: Supports PDF, DOCX, TXT, and CSV file formats.
- URL Support: Fetches content from provided URLs.
- Conversation History: Maintains a conversation history for better context.
- Gradio Interface: Provides a user-friendly interface for interaction.

## Requirements

- Python 3.6 or later
- Dependencies can be installed using `pip install -r requirements.txt`

## Usage

1. Set up your `OpenAI API` key by replacing the placeholder in the script:

   ```bash
   os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

2. Run the application by executing the `langchain_multi_document-url_chatbot.py` file:

   ```bash
   !langchain_multi_document-url_chatbot.py

3. Access the Gradio interface in your web browser by navigating to the provided URL.

3. Upload PDF, DOCX, TXT using browse file button or Enter URL using URL textbox. 

4. Enter your question in question textbox.

5. Answer will be in chatbot box.

# Contact
If you have any questions or feedback, feel free to contact me at robertselvam2002@gmail.com

## Screenshot
![screenshot](django (1).png)
