import streamlit as st
import joblib
import re
import nltk
import os
from nltk.corpus import stopwords
import pandas as pd

# Download NLTK data if not already present
@st.cache_data
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    return set(stopwords.words('english'))

# Load the model and vectorizer
@st.cache_resource
def load_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "best_model.joblib")
    vectorizer_path = os.path.join(script_dir, "tfidf_vectorizer.joblib")
    
    # Check if model files exist
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.error("🚨 Model files not found! Please run the Jupyter notebook first to train the models.")
        st.info("📝 To use this app: Run `fakenewsdetection.ipynb` to generate the required model files.")
        st.stop()
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def preprocess_text(text, stop_words):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def main():
    st.set_page_config(
        page_title="Fake News Detection",
        page_icon="📰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .fake-news {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #ef5350;
    }
    .real-news {
        background-color: #e8f5e8;
        color: #2e7d32;
        border: 2px solid #66bb6a;
    }
    .info-box {
        background-color: #ffffff;
        color: #2c3e50;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #bdc3c7;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-box strong {
        color: #1f77b4;
        font-weight: 600;
    }
    /* Ensure good contrast in dark mode */
    @media (prefers-color-scheme: dark) {
        .info-box {
            background-color: #2b2b2b;
            color: #ffffff;
            border: 1px solid #444444;
        }
        .info-box strong {
            color: #4CAF50;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">📰 Fake News Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About This App")
        st.markdown("""
        This app uses machine learning to classify news headlines as **Real** or **Fake**.
        
        **Features:**
        - Advanced text preprocessing
        - TF-IDF vectorization
        - Random Forest classifier
        - Real-time prediction
        
        **How to use:**
        1. Enter a news headline in the text area
        2. Click "Analyze News"
        3. View the prediction result
        """)
        
        st.header("📊 Model Performance")
        st.markdown("""
        The model was trained on multiple datasets including:
        - PolitiFact dataset
        - GossipCop dataset
        
        **Model Metrics:**
        - High accuracy across multiple algorithms
        - Optimized with hyperparameter tuning
        - Best performing: Random Forest
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 Enter News Headline for Analysis")
        
        # Text input
        news_text = st.text_area(
            "Enter the news headline you want to analyze:",
            height=100,
            placeholder="Example: Scientists discover new planet in distant galaxy...",
            help="Enter any news headline and our AI will predict if it's real or fake news."
        )
        
        # Sample headlines for testing
        st.subheader("🎯 Try These Sample Headlines:")
        sample_headlines = [
            "Scientists discover new planet in distant galaxy with potential for life",
            "Breaking: Government announces new economic stimulus package",
            "Aliens spotted landing in Times Square, witnesses report",
            "Local man finds cure for cancer using household items",
            "Stock market reaches new record high amid economic recovery"
        ]
        
        selected_sample = st.selectbox("Or select a sample headline:", [""] + sample_headlines)
        if selected_sample:
            news_text = selected_sample
    
    with col2:
        st.header("⚡ Quick Actions")
        analyze_button = st.button("🔍 Analyze News", type="primary", use_container_width=True)
        clear_button = st.button("🗑️ Clear Text", use_container_width=True)
        
        if clear_button:
            st.rerun()
    
    # Analysis section
    if analyze_button and news_text.strip():
        with st.spinner("🤖 Analyzing the news headline..."):
            try:
                # Load model and data
                stop_words = download_nltk_data()
                model, vectorizer = load_model()
                
                # Preprocess the text
                cleaned_text = preprocess_text(news_text, stop_words)
                
                # Vectorize
                X_vec = vectorizer.transform([cleaned_text])
                
                # Make prediction
                prediction = model.predict(X_vec)[0]
                probability = model.predict_proba(X_vec)[0]
                
                # Display results
                st.header("📋 Analysis Results")
                
                if prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-box real-news">
                        ✅ REAL NEWS
                    </div>
                    """, unsafe_allow_html=True)
                    confidence = probability[1] * 100
                else:
                    st.markdown(f"""
                    <div class="prediction-box fake-news">
                        ❌ FAKE NEWS
                    </div>
                    """, unsafe_allow_html=True)
                    confidence = probability[0] * 100
                
                # Confidence score
                st.metric("Confidence Score", f"{confidence:.1f}%")
                
                # Progress bar for confidence
                st.progress(confidence/100)
                
                # Additional information
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📊 Prediction Probabilities")
                    st.metric("Real News Probability", f"{probability[1]:.3f}")
                    st.metric("Fake News Probability", f"{probability[0]:.3f}")
                
                with col2:
                    st.subheader("📝 Text Analysis")
                    st.metric("Original Text Length", f"{len(news_text)} chars")
                    st.metric("Processed Text Length", f"{len(cleaned_text)} chars")
                
                # Show processed text
                with st.expander("🔍 View Text Processing Details"):
                    st.write("**Original Text:**")
                    st.info(news_text)
                    st.write("**Processed Text:**")
                    st.success(cleaned_text)
                    st.write("**Processing Steps Applied:**")
                    st.write("• Converted to lowercase")
                    st.write("• Removed punctuation")
                    st.write("• Removed stop words")
                    st.write("• Tokenized words")
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.info("Please make sure the model files are available and try again.")
    
    elif analyze_button and not news_text.strip():
        st.warning("⚠️ Please enter a news headline to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🚀 Powered by Machine Learning | Built with Streamlit</p>
        <p>⚠️ This tool is for educational purposes. Always verify news from multiple reliable sources.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
