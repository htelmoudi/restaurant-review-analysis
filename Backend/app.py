import spacy
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
from nltk import tokenize
import langdetect
import concurrent.futures
from transformers import BertTokenizer, BertModel

# Initialize spaCy
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize models and tokenizers for sentiment and BERT embeddings
sentiment_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
sentiment_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

# Initialize summarizer pipeline
summarizer = pipeline('summarization', model="t5-small")

# Initialize BERT tokenizer and model for sentence embeddings
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Function to calculate sentence embeddings using BERT
def get_sentence_embedding(sentence):
    inputs = bert_tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

# Language detection function
def detect_language(text):
    try:
        return langdetect.detect(text)
    except:
        return 'unknown'

# Optimized sentiment analysis function using transformers
def get_sentiment_score(text):
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True)
    outputs = sentiment_model(**inputs)
    sentiment = torch.softmax(outputs.logits, dim=-1)
    positive_score = sentiment[0][1].item()
    return positive_score if positive_score > 0.5 else -1 * (1 - positive_score)

# NLP-based anomaly detection using spaCy for text structure and sentence coherence
def detect_fake_review(review, sentiment_score, rating):
    anomalies = []
    
    # 1. Mismatch between sentiment and rating
    if sentiment_score > 0.7 and rating < 3:
        anomalies.append("Positive sentiment but low rating.")
    elif sentiment_score < -0.5 and rating > 3:
        anomalies.append("Negative sentiment but high rating.")
    
    # Tokenize review using spaCy for NLP analysis
    doc = nlp(review)
    words = [token.text for token in doc if token.is_alpha]
    word_count = len(words)
    
    # 2. Detect unnatural sentence structures using dependency parsing
    unusual_deps = [token.dep_ for token in doc if token.dep_ in ['punct', 'intj']]
    
    if word_count > 50 and len(unusual_deps) / word_count > 0.25:
        anomalies.append(f"Unusual structure detected (excessive punctuation or interjections).")
    
    # 3. Lexical diversity (generic content)
    unique_words = set([token.text.lower() for token in doc if token.is_alpha])

    lexical_diversity = len(unique_words) / word_count if word_count > 0 else 0

    if lexical_diversity < 0.3 and word_count > 30:  # Ensure short reviews aren't unfairly flagged
        anomalies.append("Low lexical diversity (generic language).")
    
    # 4. Semantic similarity (detect repetitive or generic sentences)
    sentences = [sent.text for sent in doc.sents]

    sentence_embeddings = [get_sentence_embedding(sentence) for sentence in sentences]
    similarities = []
    
    for i in range(len(sentence_embeddings)):
        for j in range(i + 1, len(sentence_embeddings)):
            sim = torch.nn.functional.cosine_similarity(sentence_embeddings[i], sentence_embeddings[j],
         dim=0)
            similarities.append(sim.item())
    
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    if avg_similarity > 0.9:  # Threshold for high sentence similarity
        anomalies.append("High sentence similarity (generic/repetitive content).")
    
    # 5. Detect overly positive or exaggerated sentiment
    if sentiment_score > 0.8 and len(words) > 50 and len(sentences) > 2:  # Very positive sentiment in a long review
        # Allow reviews with detailed content to pass without flagging excessive praise
        if len(set(words)) / word_count > 0.4:  # Check lexical diversity to ensure the review contains details
            anomalies.append("Excessive praise detected (extremely positive sentiment).")
    
    # 6. Suspicious review length
    if word_count < 10 or word_count > 500:
        anomalies.append("Suspicious review length.")
    
    return anomalies

# Function to handle batch summarization and sentiment analysis
@app.route('/analyze-reviews', methods=['POST'])
def analyze_reviews():
    data = request.get_json()
    reviews_data = data.get('reviews', [])
    
    if not reviews_data:
        return jsonify({"error": "No reviews provided"}), 400
    
    real_reviews = []
    fake_reviews = []
    review_language = detect_language(" ".join([review['review'] for review in reviews_data]))  # Batch language detection
    summaries = []
    final_weighted_score = 0
    total_weight = 0
    
    # Use concurrent processing for faster sentiment/fake detection
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_review, review_data) for review_data in reviews_data]
        for future in concurrent.futures.as_completed(futures):
            review_result = future.result()
            sentiment_score = review_result['sentiment_score']
            
            if review_result['is_fake']:
                fake_reviews.append(review_result)
                weight = 0.7  # Apply less severe penalty for fake reviews (was 0.5)
            else:
                real_reviews.append(review_result)
                weight = 1.0  # Normal weight for real reviews

            if review_result['anomalies']:
                weight *= 0.75  # Apply smaller reduction for anomalies (was 0.5)
            if len(review_result['review']) < 50:
                weight *= 0.85  # Apply smaller reduction for very short reviews (was 0.7)

            # Accumulate the weighted sentiment score
            final_weighted_score += sentiment_score * weight
            total_weight += weight

    # Summarization logic
    if real_reviews and total_weight > 0:
        real_reviews_text = " ".join([review['review'] for review in real_reviews])
        sentences = tokenize.sent_tokenize(real_reviews_text)
        chunk_size = 512
        chunks = [" ".join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]
        for chunk in chunks:
            summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        
        # Calculate final rating based on the weighted sentiment score (normalized to a 5-point scale)
        new_rating = ((final_weighted_score / total_weight) + 1) * 2.25  # Normalize sentiment to 1-5 range
    else:
        summaries.append("Not enough real reviews to summarize.")
        new_rating = "Not enough real reviews to calculate a rating."
    
    # Return all reviews and their details, including fake and real reviews
    return jsonify({
        "summarized_reviews": " ".join(summaries),
        "new_rating": round(new_rating, 2) if isinstance(new_rating, float) else new_rating,
        "fake_reviews_detected": len(fake_reviews),
        "real_reviews_used": len(real_reviews),
        "fake_reviews": fake_reviews,  # Include detailed fake reviews in the response
        "real_reviews": real_reviews,  # Include detailed real reviews in the response
        "review_language": review_language
    })

# Processing each review with adjusted fake detection and sentiment
def process_review(review_data):
    review_text = review_data['review']
    rating = review_data['rating']
    
    # Get the sentiment score using NLP
    sentiment_score = get_sentiment_score(review_text)
    
    # Check if the review is fake
    anomalies = detect_fake_review(review_text, sentiment_score, rating)
    return {
        "author": review_data.get('author', 'Unknown Author'),
        "review": review_text,
        "rating": rating,
        "time": review_data.get('time', 'No time available'),
        "is_fake": len(anomalies) > 0,
        "anomalies": anomalies,
        "sentiment_score": sentiment_score  # Include sentiment score in the response
    }

if __name__ == '__main__':
    app.run(debug=True)