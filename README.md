# Text_summarization
PDF Text Summarization
PDF Text Summarization is a Python project that allows you to extract and summarize text from PDF documents. This repository contains code and resources for extracting text from PDFs and generating concise summaries using extractive text summarization techniques.

Features
Extracts text content from PDF files.
Performs text preprocessing to improve summarization quality.
Utilizes extractive summarization based on the PageRank algorithm.
Customizable parameters for controlling the summary length and content.
Dependencies
Before running the project, ensure you have the following Python libraries installed:

Python (>=3.6)
PyMuPDF (for PDF text extraction)
nltk (Natural Language Toolkit)
networkx (for graph-based summarization)
numpy (for numerical operations)
Google Colab (for the provided code, but you can adapt it to other environments)
You can install these dependencies using  "pip install PyMuPDF nltk networkx numpy"

Additionally, for the code provided to work in Google Colab, you may need to set up Google Drive integration by mounting your Google Drive to the Colab environment. The code snippet for mounting Google Drive is as follows:
from google.colab import drive
drive.mount('/content/drive')

Usage
Clone the Repository:
Clone this repository to your local machine or your preferred environment.
git clone https://github.com/your-username/text-summarization.git

Install Dependencies:
Install the required Python libraries as mentioned in the "Dependencies" section.

Run the Summarization Script:
Use the provided Python script to summarize a PDF document. Replace your_pdf_file.pdf with the path to your PDF file.
python summarize_pdf.py --pdf_path /content/drive/MyDrive/your-folder/your_pdf_file.pdf
Customize Summarization:
You can customize the summarization process by adjusting parameters in the script, such as the number of sentences in the summary or the PageRank algorithm's parameters.

View the Summary:
The script will generate a summary of the PDF document's content and display it in the console. You can also save the summary to a file if desired.