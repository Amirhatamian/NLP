# NLP
1.Geographic_non_GeographicText_Classification
# Geographic_non_GeographicText_Classification

## Project Description
This project aims to classify text documents into two categories: geographic and non-geographic. The solution involves extracting text from various sources (including PDFs), preprocessing the text using natural language processing techniques, and then using a simple Naive Bayes classifier to make the classification.

## Technologies Used
- Python
- NLTK for text processing
- scikit-learn for machine learning
- PyPDF2 for PDF text extraction
- Google Colab for running the notebook (optional)

## Project Structure
- `notebooks/Geographic_Text_Classification.ipynb`: The main Jupyter notebook with code, explanations, and results.
- `data/`: Directory to store any sample data files (if applicable).
- `src/`: Directory containing Python scripts for modularized code.
  - `preprocessing.py`: Functions for text preprocessing.
  - `classifier.py`: The classifier and related functions.
- `requirements.txt`: List of dependencies.

## Setup Instructions
1. Clone the repository:
    ```bash
    git clone https://github.com/Amirhatamian/The Classification of Texts using Wikipedia.ipynb.git
    cd The Classification of Texts using Wikipedia.ipynb
    ```
2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Jupyter notebook:
    ```bash
    jupyter notebook notebooks/The Classification of Texts using Wikipedia.ipynb.ipynb
    ```

## How to Run the Project
1. Ensure you have all dependencies installed as per `requirements.txt`.
2. Open and run the notebook in Google Colab or Jupyter Notebook.
3. Follow the steps in the notebook to preprocess text, train the classifier, and classify new documents.

## Description of the Solution
The solution consists of the following steps:

1. **Text Extraction**: Extract text from Wikipedia articles and PDF documents.
2. **Preprocessing**: Clean and preprocess the text using techniques like tokenization, stopword removal, stemming/lemmatization.
3. **Feature Extraction**: Use CountVectorizer to convert text into numerical features.
4. **Model Training**: Train a Naive Bayes classifier on the preprocessed text data.
5. **Prediction**: Use the trained model to classify new text documents.

## Example Usage
1. Open the Jupyter notebook.
2. Run the cells to preprocess the text, train the classifier, and classify the PDF document.

If you encounter any issues or have any questions, please open an issue in this repository.
