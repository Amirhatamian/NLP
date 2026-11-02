# NLP: Text Classification and Summarisation with Wikipedia

Two small natural language processing projects built on text taken from Wikipedia.

## 1. Geographic vs non-geographic text — `The_Classification_of_Texts_using_Wikipedia.ipynb`

**Goal:** decide whether a document is about a place (geographic) or not, and use the model to classify text extracted from a PDF.

**What I did**
- Downloaded Wikipedia summaries for 10 topics: 5 geographic (New York, Mount Everest, Sahara Desert, Amazon River, Paris) and 5 non-geographic (Quantum Mechanics, Artificial Intelligence, Shakespeare, Modern Art, Jazz Music).
- Pre-processed the text with NLTK: tokenisation, part-of-speech tagging, stop-word removal, lemmatisation, keeping only nouns.
- Found the 10 most frequent nouns for each class (geographic: *river, city, desert, region…*; non-geographic: *theory, art, system, intelligence…*) and used these 20 words as features (word counts).
- Trained a logistic regression model and compared it with a majority-class baseline.
- Wrote a function that reads a PDF with PyPDF2 and classifies its text.

**Result and limits:** this is a working prototype of the full pipeline. With only 9 documents downloaded, the test set has 2 documents, so the 50% test accuracy says little about the method. The next step is to train on a few hundred Wikipedia pages per class.

## 2. Hierarchical summarisation for a language model's context window — `LLM.ipynb`

**Goal:** shorten two long texts so that both fit in a language model's context window, keeping their lengths in proportion.

**What I did**
- Downloaded two Wikipedia articles chosen by the user and saved them as text files.
- Calculated a target length for each text from its original length and the context window size (128 words in the example).
- **Version 1:** cut the text into slices, shortened each slice by keeping its first and last sentences, joined the results, and repeated until the text fitted the window.
- **Version 2:** picked sentences by similarity instead: built a sentence similarity matrix (word counts + cosine similarity), started from the first sentence and kept adding the sentence most similar to the summary so far, until the target length was reached.
- Combined the two summaries into one prompt for the language model.

## How to run

```bash
git clone https://github.com/Amirhatamian/NLP.git
cd NLP
pip install nltk scikit-learn wikipedia wikipedia-api PyPDF2 jupyter
jupyter notebook
```

Both notebooks download their data from Wikipedia, so they need an internet connection.

## Tools

Python, NLTK, scikit-learn, PyPDF2, Wikipedia API
