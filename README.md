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

#MLSDL

## Problem framing
Business objective: Build a customer support chatbot that answers common questions about passwords, shipping, returns, payments, and contact info.
ML task: kNN classification—match user questions to the most relevant predefined answer using nearest neighbor search on embeddings.

## Hypothesis formulation
Expected patterns: User questions about the same topic (e.g., "password reset," "forgot login") will cluster together in embedding space.
Predictive signals: Measuring the distance between the vectors & deciding that < 0.3 indicates good match.

## Data collection and understanding
Data sources: 5 predefined Q&A pairs covering common customer support topics.
Structure: Simple list of strings; German-focused customer service domain.
Limitations: Small dataset (5 documents), no training data, potential for out-of-domain questions.

## Exploratory data analysis (minimal)
Approach: Manual check of embedding distances between known similar questions.
Findings: Baseline questions are well-separated; threshold of 0.3 works for distinguishing good vs poor matches.
Risks: Limited coverage means weak matches for edge cases.

## Hypothesis validation before modeling
Simple rule: Test that known-similar questions have cosine distance < 0.3.
Baseline check: Keyword matching fails on paraphrases (distance > 0.6), confirming kNN adds value.

## Feature engineering
Raw → features: Convert text questions to 384-dim vectors using all-MiniLM-L6-v2.
No additional: Pretrained embeddings handle encoding/scaling automatically.

## Model development
Baseline: Exact string matching (rejected—poor paraphrase handling).
Main approach: kNN with scikit-learn NearestNeighbors (k=3, cosine metric).
Why kNN: No training needed, scales with documents, interpretable distances.

## Evaluation
Metrics: Cosine distance (lower = better match); business goal = relevant answers > 70% of time.
Validation: The 0.3 threshold was hand-tuned for this small demo dataset between the question and the closest knowledge match.

## Deployment (Next step)
Current: CLI chat loop (main.py).
Production path: Wrap get_best_answer() in FastAPI; log confidence scores for monitoring.

## Optimization (Next step)
Monitoring: Track average match confidence, unanswered question rate, user feedback.
Next steps: Expand knowledge base from real queries; tune k/threshold; add human fallback for low-confidence cases.
