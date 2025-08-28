# app.py
import streamlit as st
import joblib
import xgboost as xgb
import pandas as pd
from scipy.sparse import hstack
from textblob import TextBlob  # For sentiment analysis

# Load model components
@st.cache_resource
def load_models():
    model = joblib.load('models/toxicity_xgboost_model.pkl')
    tfidf = joblib.load('models/tfidf_vectorizer.pkl')
    scaler = joblib.load('models/scaler.pkl')
    numerical_features = joblib.load('models/numerical_features.pkl')
    return model, tfidf, scaler, numerical_features

model, tfidf, scaler, numerical_features = load_models()

# Function to calculate sentiment polarity
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Function to estimate toxic word ratio
def estimate_toxic_ratio(text):
    toxic_words = ['idiot', 'stupid', 'moron', 'kill', 'die', 'shut up', 'hate', 'dumb', 'ugly', 'trash']
    words = text.lower().split()
    toxic_count = sum(1 for word in words if word in toxic_words)
    return toxic_count / len(words) if words else 0

# App UI
st.title("🛡️ Toxicity Detection App")
st.markdown("Enter a comment below to check if it's toxic.")

# Input form
with st.form("toxicity_form"):
    comment = st.text_area("Enter your comment:", "This is a sample comment")
    submitted = st.form_submit_button("Check Toxicity")

# Prediction function
def predict_toxicity(comment_text):
    # Calculate additional features
    word_count = len(comment_text.split())
    sentiment = get_sentiment(comment_text)
    toxic_ratio = estimate_toxic_ratio(comment_text)

    # Create DataFrame
    df = pd.DataFrame({
        'normalized_text': [comment_text],
        'comment_length': [len(comment_text)],
        'word_count': [word_count],
        'sentiment_polarity': [sentiment],
        'word_density': [word_count / len(comment_text)],
        'toxic_word_ratio': [toxic_ratio]
    })

    # Preprocess
    X_text = tfidf.transform(df['normalized_text'])
    X_num = scaler.transform(df[numerical_features])
    X_new = hstack([X_num, X_text])

    # Predict
    dnew = xgb.DMatrix(X_new)
    prob = model.predict(dnew)[0]
    prediction = 1 if prob > 0.5 else 0

    return prob, prediction

# Display results
if submitted and comment.strip():
    prob, prediction = predict_toxicity(comment)

    # Results
    st.subheader("Prediction Result")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Prediction", "TOXIC" if prediction else "NON-TOXIC")
    with col2:
        st.metric("Toxicity Probability", f"{prob:.2f}")

    # Visual indicator
    st.progress(float(prob))
    st.caption(f"Toxicity confidence: {prob:.2%}")

    # Interpretation
    if prediction:
        st.warning("⚠️ This comment is predicted to be toxic.")
    else:
        st.success("✅ This comment appears to be non-toxic.")

    # Show feature details
    with st.expander("See feature details"):
        st.write(f"**Word Count:** {len(comment.split())}")
        st.write(f"**Sentiment Polarity:** {get_sentiment(comment):.2f}")
        st.write(f"**Estimated Toxic Word Ratio:** {estimate_toxic_ratio(comment):.2f}")
else:
    st.info("Please enter a comment to analyze.")
