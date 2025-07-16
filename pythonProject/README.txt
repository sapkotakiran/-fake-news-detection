# 📰 Fake News Detection System

A comprehensive machine learning project that detects fake news using natural language processing techniques and multiple classification algorithms.

## 🎯 Project Overview

This project implements a fake news detection system using machine learning algorithms to classify news headlines as either "Real" or "Fake". The system is trained on multiple datasets and provides both a web API (Flask) and an interactive web interface (Streamlit) for real-time predictions.

## 📊 Dataset Information

The project uses four CSV datasets containing news articles with their labels:

- **`politifact_fake.csv`** - Fake news articles from PolitiFact
- **`politifact_real.csv`** - Real news articles from PolitiFact  
- **`gossipcop_fake.csv`** - Fake news articles from GossipCop
- **`gossipcop_real.csv`** - Real news articles from GossipCop

**Dataset Statistics:**
- Combined dataset contains thousands of news headlines
- Balanced using SMOTE (Synthetic Minority Oversampling Technique)
- Labels: 0 = Fake News, 1 = Real News

## 🏗️ Project Structure

```
pythonProject/
├── README.md                    # Project documentation
├── app.py                      # Flask API application
├── streamlit_app.py            # Streamlit web interface
├── fakenewsdetection.ipynb     # Jupyter notebook with analysis
├── combined_dataset.csv        # Processed combined dataset
├── best_model.joblib           # Trained Random Forest model
├── best_svm_model.joblib       # Trained SVM model (alternative)
├── tfidf_vectorizer.joblib     # TF-IDF vectorizer
├── politifact_fake.csv         # PolitiFact fake news data
├── politifact_real.csv         # PolitiFact real news data
├── gossipcop_fake.csv          # GossipCop fake news data
├── gossipcop_real.csv          # GossipCop real news data
└── data/                       # Additional data directory
```

## 🔧 Technical Implementation

### Data Preprocessing
- **Text Cleaning**: Lowercase conversion, punctuation removal
- **Stop Words Removal**: Common English words filtered out
- **Tokenization**: Text split into individual words
- **TF-IDF Vectorization**: Convert text to numerical features (max 5000 features)

### Machine Learning Models Tested
1. **Logistic Regression** - Simple baseline model
2. **Naive Bayes** - Probabilistic classifier for text
3. **Support Vector Machine (SVM)** - High-dimensional classification
4. **Random Forest** ⭐ - Best performing ensemble method
5. **XGBoost** - Gradient boosting algorithm
6. **Hyperparameter Tuned Models** - Optimized versions

### Model Performance
The **Random Forest** classifier achieved the best performance:
- **High Accuracy**: Consistently performs well across metrics
- **Robust**: Less prone to overfitting
- **Feature Importance**: Provides insights into important words
- **Balanced Dataset**: Trained on SMOTE-balanced data

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### 1. Clone/Download the Project
```bash
git clone <repository-url>
cd "Fake news Detection/pythonProject"
```

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn nltk joblib flask streamlit
pip install matplotlib seaborn xgboost imbalanced-learn
```

### 3. Download NLTK Data
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

## 🖥️ Usage

### Option 1: Streamlit Web Interface (Recommended)
```bash
streamlit run streamlit_app.py
```
- **URL**: http://localhost:8501
- **Features**: Interactive UI, sample headlines, confidence scores
- **Best for**: Demonstrations, testing, user-friendly interface

### Option 2: Flask API
```bash
python app.py
```
- **URL**: http://localhost:5000
- **API Endpoint**: POST `/predict`
- **Best for**: Integration with other applications

#### API Usage Example:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "Scientists discover new planet in distant galaxy"}'
```

Response:
```json
{"prediction": "Real"}
```

### Option 3: Jupyter Notebook
```bash
jupyter notebook fakenewsdetection.ipynb
```
- **Best for**: Data analysis, model training, experimentation

## 📱 Web Interface Features

### Streamlit App Highlights:
- 🎨 **Modern UI**: Professional design with color-coded results
- 📊 **Confidence Scores**: Shows prediction probability
- 🎯 **Sample Headlines**: Pre-loaded examples for testing
- 📈 **Model Info**: Performance metrics and dataset information
- 🔍 **Text Analysis**: Shows preprocessing steps
- 📱 **Responsive**: Works on desktop and mobile

## 🔬 Model Training Process

1. **Data Loading**: Load and combine four CSV datasets
2. **Data Cleaning**: Remove null values, preprocess text
3. **Feature Engineering**: TF-IDF vectorization
4. **Class Balancing**: Apply SMOTE for balanced training
5. **Model Training**: Train multiple algorithms
6. **Hyperparameter Tuning**: Grid search optimization
7. **Model Evaluation**: Compare performance metrics
8. **Model Selection**: Choose best performing model
9. **Model Saving**: Export trained model and vectorizer

## 📈 Evaluation Metrics

The models are evaluated using:
- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed prediction breakdown

## 🔍 Key Features

### Text Preprocessing Pipeline:
1. Convert to lowercase
2. Remove punctuation and special characters
3. Remove stop words (the, and, is, etc.)
4. Tokenize into words
5. Apply TF-IDF vectorization

### Model Capabilities:
- Real-time prediction
- Confidence scoring
- Handles various text formats
- Robust to different writing styles

## 🛠️ Development

### Adding New Features:
1. **New Models**: Add to `fakenewsdetection.ipynb`
2. **API Endpoints**: Extend `app.py`
3. **UI Components**: Modify `streamlit_app.py`
4. **Data Sources**: Update data loading functions

### File Descriptions:
- **`app.py`**: Flask REST API with prediction endpoint
- **`streamlit_app.py`**: Interactive web interface
- **`fakenewsdetection.ipynb`**: Complete analysis and training
- **`*.joblib`**: Serialized models and vectorizer
- **`combined_dataset.csv`**: Preprocessed training data

## ⚠️ Important Notes

### Model Limitations:
- Trained on specific news domains (PolitiFact, GossipCop)
- Performance may vary on different news sources
- Requires headlines in English
- Best results with news-style text

### Recommendations:
- Always verify predictions with multiple sources
- Use as a screening tool, not definitive judgment
- Regularly retrain with new data
- Consider context and source credibility

## 🔧 Configuration

### Model Parameters:
- **TF-IDF**: Max features = 5000
- **Random Forest**: n_estimators = 100
- **Train/Test Split**: 80/20
- **Cross Validation**: 5-fold

### Environment Variables:
- No special configuration required
- Models load automatically from script directory

## 📊 Performance Insights

### Best Model Analysis:
- **Random Forest** chosen for final deployment
- Excellent performance on balanced dataset
- Good generalization across news types
- Interpretable feature importance

### Training Data:
- Combines multiple reputable fact-checking sources
- Balanced fake/real distribution after SMOTE
- Diverse news topics and writing styles

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is for educational purposes. Please cite appropriately if used in research.

## 🙏 Acknowledgments

- **PolitiFact** and **GossipCop** for datasets
- **scikit-learn** for machine learning tools
- **Streamlit** for web interface framework
- **Flask** for API development

## 📞 Support

For questions or issues:
1. Check existing documentation
2. Review Jupyter notebook for detailed analysis
3. Test with provided sample data
4. Verify all dependencies are installed

---

**⚡ Quick Start:**
```bash
# Install dependencies
pip install streamlit pandas scikit-learn nltk joblib

# Run the app
streamlit run streamlit_app.py

# Open browser to http://localhost:8501
```

**🎯 Try it now with sample headlines:**
- "Scientists discover new planet in distant galaxy with potential for life" (Real)
- "Aliens spotted landing in Times Square, witnesses report" (Fake)

---

*Built with ❤️ using Python, scikit-learn, and Streamlit*
