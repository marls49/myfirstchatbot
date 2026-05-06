# kNN Embedding Chatbot

This was the first chatbot that I have coded myself - done as part of a deep learning course. 

I love it as it simply says the accuracy when it doesn't understand something - maybe more of a joke for a budding software developer than a customer ;-) 

It uses Transformers too - something of a personal goal since ChatGPT. 

A simple semantic retrieval chatbot built with Sentence Transformers and scikit-learn.

## Project structure

- `config.py` stores documents, answers, and runtime settings.
- `embeddings.py` loads the embedding model and creates vector embeddings.
- `index.py` builds the nearest-neighbor index.
- `retrieval.py` runs semantic search and converts distance to similarity.
- `chatbot.py` combines the pieces into a reusable chatbot class.
- `main.py` runs the interactive CLI.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

Type a question and the bot returns the closest matching answer. If the match is weak, it asks you to rephrase.

## How it works

1. Questions are embedded with `all-MiniLM-L6-v2`.
2. The embeddings are indexed with cosine-distance kNN.
3. The closest document is selected.
4. If the distance is above the threshold, the bot tells you what the prediction is and that is it too low to answer.

## Notes

You can extend this by loading documents from a file, separating data from code, or adding a feedback loop for unanswered questions.
