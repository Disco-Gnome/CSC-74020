# Project Title: PDF-Lucy

The purpose of this repo is to have starter code for students to build a Python Streamlit application using the power of GenAI to analyze and chat with .pdf files.

## Features
- Process .pdf files using different approaches (selected by user)
- Easy setup and use

## Installation and Setup

### Prerequisites

You need to have Python 3.8 or later to use this application. If you don't have Python installed, you can download it from the official site: https://www.python.org/downloads/

### Steps

1. Clone the repository to your local machine.

   ```bash
   git clone https://github.com/Markovian99/CSC-74020.git
   ```
   
2. Navigate to the within the PDFLucy directory.

   ```bash
   cd PDFLucy
   ```
   
3. Install the necessary packages using pip. (Recommend using a virtual environment)

   ```bash
   pip install -r requirements.txt
   ```
   
4. In the project's root directory, create a `.env` file to store your API keys securely.
   
5. Open the `.env` file using any text editor and enter your API keys as shown below (if using OpenAI):

   ```bash
   export GOOGLE_GEMINI_API_KEY = "YOUR GOOGLE GEMINI KEY"
   export ANTHROPIC_API_KEY = "YOUR ANTROPIC API KEY"
   export OPENAI_API_KEY = "YOUR OPENAI API KEY"
   ```
   If using Azure for OPEN AI, also include
   ```bash
   export OPENAI_API_BASE = "YOUR OPENAI API BASE"
   export OPENAI_API_TYPE = "YOUR OPENAI API TYPE"
   export OPENAI_API_VERSION = "YOUR OPENAI API VERSION"
   ```

## Usage

To run the PDFLucy application:

```bash
cd src
streamlit run app.py
```

## License

This project is licensed under the terms of the Apache License 2.0. For more details, please see the [LICENSE](LICENSE) file.

## Support

For any questions or issues, please contact the maintainers, or raise an issue in the GitHub repository.
