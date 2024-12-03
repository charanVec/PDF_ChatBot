# PDF_ChatBot

This repository contains a PDF-based customer chatbot designed to help users study and interact with large PDF documents efficiently. The application leverages **Machine Learning (ML)** and **Deep Learning (DL)** techniques along with **LangChain API integration** to deliver seamless interactions.

---

## Features

### 1. **Home Page**
- **Upload Multiple PDFs**: Easily upload multiple files to the system.
- **Optimized File Storage**: Files are stored in a designated folder to minimize local system space usage.
- **Heavy PDF Support**: Capable of handling large PDF files (including PDFs with up to 70 billion words). Note that large files may take time to load.

### 2. **Query Page**
- **Query Box**: Input your questions directly related to the uploaded PDFs.
- **Answer Generation**: Click the **"Ask"** button to generate contextually relevant answers based on the content of the PDFs.
- **Chat History**: Maintain a record of questions and answers for quick reference.
- **Copy Functionality**: Copy answers directly for use elsewhere.
- **Clear Button**: Quickly clear the chat history.

---

## Technology Stack
- **Backend**: Flask (Python)
- **Machine Learning**: Used for text processing and optimization.
- **Deep Learning**: Enhanced text comprehension and response generation.
- **LangChain API**: For connecting and interacting with PDFs.
- **Frontend**: HTML, CSS, and JavaScript for building interactive pages.

---

## Key Functionalities
1. **Efficient PDF Parsing**: Supports parsing and processing of large PDF documents.
2. **Interactive Chatbot**: Provides an intuitive interface for interacting with uploaded files.
3. **File Management**: Stores uploaded files externally to ensure efficient memory usage.
4. **Scalable Design**: Handles PDFs with very high word counts for extensive data exploration.

---

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/charanVec/PDF_ChatBot.git
   cd PDF_ChatBot
   ```

2. **Install Required Dependencies**
   Use a Python virtual environment (optional but recommended).
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up LangChain API Key**
   - Obtain an API key from LangChain.
   - Add the key to the environment variables or `config.py`.

4. **Run the Application**
   ```bash
   python app.py
   ```

5. **Access the Application**
   - Open a browser and go to `http://127.0.0.1:5000`.

---

## Usage
1. **Navigate to the Home Page**:
   - Upload one or more PDF files.
2. **Switch to the Query Page**:
   - Ask questions related to the uploaded PDFs.
   - View, copy, or clear the chat history as needed.

---

## Limitations
- Large PDF files may take longer to process due to their size.
- Requires an active internet connection for LangChain API integration.

---
## Used By
![](https://github.com/prabhakarvenkat/Potential_Customer_for_upsell/blob/4f1ba9a3b0e251d4a482cff767e11512e7c07e6d/assets/cognizant.jpg)

---
## Contribution
Feel free to contribute to this project by forking the repository and submitting pull requests. Issues and feature requests are welcome.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
