# E-mail Spam Detection Project

A machine learning-based email spam detection system that classifies emails as spam or legitimate messages (ham) using multiple classification algorithms and natural language processing techniques.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Models and Techniques](#models-and-techniques)
- [Installation](#installation)
- [Usage](#usage)
- [Results and Performance](#results-and-performance)
- [File Descriptions](#file-descriptions)
- [Technical Implementation](#technical-implementation)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project develops and evaluates multiple machine learning models to automatically detect spam emails. Email spam is a persistent problem in digital communication, and automated detection systems are essential to filter unwanted messages. This project implements a comprehensive solution using state-of-the-art machine learning techniques.

### Problem Statement

The objective is to build a binary classification system that can accurately distinguish between:
- **Ham (0)**: Legitimate emails
- **Spam (1)**: Unsolicited or malicious emails

### Approach

The project follows a systematic machine learning pipeline:
1. Data Loading and Exploration
2. Data Preprocessing and Text Cleaning
3. Feature Extraction using Vectorization Techniques
4. Model Training and Evaluation
5. Model Selection and Deployment
6. Prediction on New Emails

## Project Structure

```
E-mail-Spam-Detection-Project/
│
├── E-mail Spam detection/
│   ├── Project.ipynb                                  # Main Jupyter notebook with complete implementation
│   ├── spam.csv                                       # Dataset containing 5,572 emails
│   ├── email_spam_detection.pkl                       # Trained Logistic Regression model
│   ├── email_spam_detection_logistic_regression.pkl   # Alternative trained model (Multinomial Naive Bayes)
│   ├── requirements.txt                               # Python dependencies
│   └── .ipynb_checkpoints/                            # Jupyter notebook checkpoints
│
└── README.md                                          # Project documentation

```

## Dataset

### Source
The project uses the **UCI Machine Learning Repository Spam Dataset**, a widely-used benchmark dataset for email classification tasks.

### Dataset Statistics
- **Total Emails**: 5,572 messages
- **Features**: 2 main columns
  - `v1`: Label (ham or spam)
  - `v2`: Email message text
- **Class Distribution**:
  - Ham (Legitimate): Majority class (~4,827 emails)
  - Spam: Minority class (~745 emails)
  - Class Imbalance Ratio: ~6.5:1

### Data Format
- **Encoding**: ISO-8859-1
- **Format**: CSV with comma-separated values
- **File Size**: spam.csv

### Data Characteristics
- Raw email text with varied formatting
- Mix of SMS messages and email content
- Contains special characters, numbers, and symbols
- Real-world noisy data with typos and abbreviations

## Models and Techniques

### 1. Feature Extraction Methods

#### CountVectorizer
- Converts text to a matrix of token counts
- Creates a vocabulary of all tokens in the corpus
- Represents each email as a vector of word frequencies
- Used in the baseline Logistic Regression model

#### TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer
- Advanced text representation technique
- `sublinear_tf=True`: Applies sublinear term frequency scaling
- Down-weights frequently occurring terms across documents
- Better captures the importance of unique terms
- Used for training multiple advanced models

### 2. Classification Algorithms

#### a) Logistic Regression
- **Type**: Linear classification model
- **Advantages**: Fast, interpretable, efficient
- **Use Case**: Baseline model for comparison
- **Vectorizer**: CountVectorizer
- **Performance**: Accuracy reported in results section

#### b) Multinomial Naive Bayes
- **Type**: Probabilistic classifier based on Bayes' theorem
- **Advantages**: Works well with text data, fast training
- **Assumption**: Conditional independence of features
- **Vectorizer**: TF-IDF Vectorizer
- **Best for**: Text classification tasks

#### c) Support Vector Machine (SVM)
- **Type**: Discriminative classifier
- **Advantages**: Robust to high-dimensional data, effective margin maximization
- **Vectorizer**: TF-IDF Vectorizer
- **Best for**: Binary classification with clear decision boundaries

#### d) Random Forest Classifier
- **Type**: Ensemble method using decision trees
- **Advantages**: Handles non-linear relationships, reduces overfitting
- **Ensemble Size**: Multiple decision trees
- **Vectorizer**: TF-IDF Vectorizer
- **Best for**: Complex patterns and interactions

#### e) XGBoost (eXtreme Gradient Boosting)
- **Type**: Gradient boosting ensemble method
- **Advantages**: High performance, handles imbalanced data well
- **Vectorizer**: TF-IDF Vectorizer
- **Best for**: Achieving state-of-the-art results

### 3. Text Preprocessing

#### Tokenization
- Uses `RegexpTokenizer` from NLTK library
- Pattern: `r'[a-zA-Z0-9]'` to extract alphanumeric tokens
- Removes special characters, punctuation, and whitespace
- Creates meaningful token sequences

#### Label Encoding
- Converts categorical labels (ham/spam) to numerical values
- Ham → 0, Spam → 1
- Uses `LabelEncoder` from scikit-learn

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Chakrapani2122/E-mail-Spam-Detection-Project.git
cd E-mail-Spam-Detection-Project
```

### Step 2: Create Virtual Environment (Optional but Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
cd "E-mail Spam detection"
pip install -r requirements.txt
```

### Key Dependencies
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms and utilities
- **nltk**: Natural language processing toolkit
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **xgboost**: Gradient boosting framework
- **jupyter**: Interactive notebook environment
- **keras** & **tensorflow**: Deep learning (included in requirements)
- **Flask** (optional): For web deployment

## Usage

### 1. Running the Full Pipeline

Open and run the Jupyter notebook:
```bash
cd "E-mail Spam detection"
jupyter notebook Project.ipynb
```

The notebook will:
- Load and explore the dataset
- Preprocess and clean the text data
- Train multiple classification models
- Evaluate model performance
- Save trained models to pickle files

### 2. Using Pre-trained Models

#### Loading a Saved Model
```python
import pickle

# Load the Logistic Regression model
model = pickle.load(open('email_spam_detection.pkl', 'rb'))

# Load the Multinomial Naive Bayes model
model = pickle.load(open('email_spam_detection_logistic_regression.pkl', 'rb'))
```

#### Making Predictions on New Emails
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize vectorizer (or load if saved)
vectorizer = TfidfVectorizer(sublinear_tf=True, encoding='utf-8', decode_error='ignore')

# Example emails to classify
example_messages = [
    "Congratulations, you've won a prize!",  # Likely spam
    "Hello, how are you doing today?"         # Likely ham
]

# Vectorize the messages
messages_transformed = vectorizer.transform(example_messages)

# Make predictions
predictions = model.predict(messages_transformed)

# Display results
for message, prediction in zip(example_messages, predictions):
    print(f"Message: {message}")
    print(f"Predicted: {'SPAM' if prediction == 1 else 'HAM'}")
    print()
```

### 3. Training Your Own Model

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv('spam.csv', encoding='ISO-8859-1')
df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace=True)

# Encode labels
df['v1'] = df['v1'].map({'ham': 0, 'spam': 1})

# Vectorize text
vectorizer = TfidfVectorizer(sublinear_tf=True, encoding='utf-8', decode_error='ignore')
X = vectorizer.fit_transform(df['v2'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, df['v1'], test_size=0.25, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions)}")
print(classification_report(y_test, predictions))
```

## Results and Performance

### Model Evaluation Metrics

The project evaluates models using:
- **Accuracy**: Percentage of correct classifications
- **Confusion Matrix**: True Positives, True Negatives, False Positives, False Negatives
- **Classification Report**: Precision, Recall, and F1-Score for each class

### Expected Performance Range
- **Logistic Regression**: ~97% accuracy with CountVectorizer
- **Multinomial Naive Bayes**: ~97-98% accuracy with TF-IDF
- **SVM**: High accuracy with proper kernel selection
- **Random Forest**: ~96-97% accuracy
- **XGBoost**: ~97-98% accuracy (often best performing)

### Key Performance Indicators
- **Precision**: Important for minimizing false spam alerts (legitimate emails marked as spam)
- **Recall**: Important for catching actual spam messages
- **F1-Score**: Harmonic mean balancing precision and recall

### Confusion Matrix Interpretation
```
                 Predicted
                 Spam    Ham
Actual  Spam      TP      FN
        Ham       FP      TN
```
- True Positives (TP): Correctly identified spam
- True Negatives (TN): Correctly identified ham
- False Positives (FP): Legitimate emails marked as spam (Type I error)
- False Negatives (FN): Spam missed as legitimate (Type II error)

## File Descriptions

### Project.ipynb
**Type**: Jupyter Notebook  
**Purpose**: Main implementation file containing the complete machine learning pipeline

**Contents**:
1. **Import Section**: Loads all required libraries
   - Data processing: pandas, numpy
   - ML algorithms: scikit-learn, xgboost, keras
   - Visualization: matplotlib, seaborn
   - NLP: nltk

2. **Data Loading**: Reads spam.csv dataset

3. **Data Exploration**: 
   - Dataset shape and size information
   - Column names and data types
   - Null value checks
   - Class distribution visualization

4. **Data Preprocessing**:
   - Drops unnecessary columns
   - Encodes categorical labels
   - Tokenizes email text

5. **Feature Extraction**:
   - CountVectorizer for initial model
   - TF-IDF Vectorizer for advanced models

6. **Model Training**:
   - Logistic Regression with CountVectorizer
   - Multinomial Naive Bayes with TF-IDF
   - SVM with TF-IDF
   - Random Forest with TF-IDF
   - XGBoost with TF-IDF

7. **Model Evaluation**:
   - Confusion matrices
   - Classification reports
   - Accuracy scores

8. **Model Persistence**:
   - Saves trained models as pickle files
   - Loads models for inference
   - Tests with example emails

### spam.csv
**Type**: Comma-Separated Values  
**Purpose**: Main dataset for training and testing

**Columns**:
- `v1`: Label (ham or spam)
- `v2`: Email/SMS text content
- Unnamed: 2, 3, 4: Empty columns (dropped during preprocessing)

**Usage**: Training data source for all machine learning models

### email_spam_detection.pkl
**Type**: Pickled Python Object  
**Purpose**: Serialized Logistic Regression model trained with CountVectorizer

**Usage**:
```python
import pickle
model = pickle.load(open('email_spam_detection.pkl', 'rb'))
predictions = model.predict(X_test)
```

### email_spam_detection_logistic_regression.pkl
**Type**: Pickled Python Object  
**Purpose**: Serialized Multinomial Naive Bayes model trained with TF-IDF

**Usage**: Production inference on new emails

### requirements.txt
**Type**: Text file  
**Purpose**: Lists all Python package dependencies

**Key Packages**:
- Machine Learning: scikit-learn, xgboost, keras, tensorflow
- Data Processing: pandas, numpy
- Visualization: matplotlib, seaborn
- NLP: nltk
- Utilities: jupyter, ipython, flask

## Technical Implementation

### Data Pipeline

```
Raw Email Text
    ↓
Text Preprocessing (Tokenization, Cleaning)
    ↓
Feature Extraction (CountVectorizer or TF-IDF)
    ↓
Feature Matrix
    ↓
Model Training
    ↓
Classification (Spam/Ham)
```

### Model Training Process

1. **Data Split**: 75% training, 25% testing (random_state=42)
2. **Feature Extraction**: Convert text to numerical features
3. **Model Fit**: Train classifier on training data
4. **Prediction**: Generate predictions on test data
5. **Evaluation**: Calculate accuracy and other metrics
6. **Serialization**: Save model using pickle

### Key Hyperparameters

- **TfidfVectorizer**:
  - `sublinear_tf=True`: Apply sublinear term frequency scaling
  - `encoding='utf-8'`: UTF-8 character encoding
  - `decode_error='ignore'`: Ignore decoding errors

- **train_test_split**:
  - `test_size=0.25`: 25% test data
  - `random_state=42`: Reproducible splits

### Text Representation

#### CountVectorizer Approach
- Creates a vocabulary dictionary
- Counts term occurrences per document
- Simple, interpretable feature representation
- Suitable for baseline models

#### TF-IDF Approach
- Term Frequency: How often a word appears in a document
- Inverse Document Frequency: How unique the word is across all documents
- Formula: TF-IDF = TF × log(N/DF)
- Better represents document importance and relevance

### Class Distribution Handling

The dataset has inherent class imbalance (6.5:1 ratio). Strategies employed:
- Metrics choice: F1-score preferred over accuracy
- Model selection: Some models handle imbalance better
- Future improvements could use SMOTE or class weights

## Contributing

We welcome contributions to improve this project! Here are ways you can contribute:

### Ways to Contribute
1. **Bug Reports**: Report issues or bugs in the code
2. **Feature Requests**: Suggest new features or improvements
3. **Code Improvements**: Submit pull requests with enhancements
4. **Documentation**: Improve or expand the documentation
5. **Model Improvements**: Experiment with better algorithms or tuning

### Contribution Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes and test thoroughly
4. Commit with clear, descriptive messages
5. Push to your branch and create a Pull Request
6. Ensure your code follows project conventions

### Areas for Improvement
- Implement deep learning models (LSTM, CNN)
- Add web interface using Flask or Django
- Implement API endpoint for predictions
- Add real-time email filtering capabilities
- Explore transfer learning with pre-trained models
- Implement advanced ensemble methods
- Add cross-validation for robust evaluation
- Deploy to cloud platforms (AWS, GCP, Azure)

## License

This project is provided as-is for educational and research purposes. Please refer to the repository for any specific license information.

### Attribution
The dataset is sourced from the UCI Machine Learning Repository:
- **Dataset**: SMS Spam Collection
- **Source**: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
- **Citation**: UCI ML Repository contributors

## Future Enhancements

1. **Deep Learning Models**: Implement LSTM or Transformer-based architectures
2. **Feature Engineering**: Extract advanced features (sender reputation, link analysis)
3. **Web Application**: Build a Flask/Django web interface for easy predictions
4. **Real-time Integration**: Connect with email providers for live filtering
5. **Model Deployment**: Deploy models to cloud platforms
6. **Cross-validation**: Implement k-fold cross-validation for robust evaluation
7. **Hyperparameter Tuning**: Use GridSearchCV or RandomizedSearchCV
8. **Multilingual Support**: Extend to detect spam in multiple languages
9. **Image-based Spam**: Add capabilities to detect spam in email images
10. **Performance Optimization**: Optimize inference speed for production use

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'sklearn'`
- **Solution**: Install scikit-learn: `pip install scikit-learn`

**Issue**: `UnicodeDecodeError` when reading spam.csv
- **Solution**: The encoding is already set to ISO-8859-1 in the code

**Issue**: Memory error with large datasets
- **Solution**: Use batching or streaming approaches for large-scale data

**Issue**: Pickle module error when loading models
- **Solution**: Ensure Python version compatibility between model training and loading

## Contact and Support

For questions, issues, or support, please:
1. Open an issue on the GitHub repository
2. Check existing issues for similar problems
3. Provide detailed information about your problem
4. Include error messages and reproduction steps

---

**Last Updated**: 2024  
**Project Status**: Active  
**Author**: Chakrapani2122


