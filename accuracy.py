import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import re
from textblob import TextBlob

# Your extract_features function (with fixes)
def extract_features(df):
    def clean_text(text):
        text = str(text) if not pd.isna(text) else ""
        text = re.sub(r'#URL_[a-f0-9]+#', '', text)
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        text = re.sub(r'[^\w\s.,!?]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()

    def get_sentiment(text):
        cleaned = clean_text(text)
        if not cleaned:
            return 0.0, 0.0, 0.0
        blob = TextBlob(cleaned)
        return blob.sentiment.polarity, blob.sentiment.subjectivity, 0.0

    def detect_scam_keywords(text):
        scam_keywords = [
            'no experience needed', 'work from home', 'easy money', 'get paid weekly',
            'limited spots', 'apply now', 'free training', 'send money', 'bank details',
            'photo id', 'western union', 'paypal', 'huge income', 'earn up to', 'instant interview',
            'no skills required', 'quick cash', 'positions fill up quick', 'pay small fee', 'recruit friends',
            'act fast', 'spots left', 'from anywhere', 'independent work', 'starter kit'
        ]
        cleaned = clean_text(text).lower()
        matches = sum(1 for keyword in scam_keywords if keyword in cleaned)
        return min(matches / len(scam_keywords), 1.0)

    features_list = []
    for _, row in df.iterrows():
        job_data = row.to_dict()
        title = clean_text(job_data.get('title', ''))
        description = clean_text(job_data.get('description', ''))
        requirements = clean_text(job_data.get('requirements', ''))
        company_profile = clean_text(job_data.get('company_profile', ''))
        combined_text = ' '.join([title, description, requirements, company_profile])

        desc_sent, desc_subj, desc_scam = get_sentiment(description)
        req_sent, req_subj, req_scam = get_sentiment(requirements)
        comp_sent, comp_subj, comp_scam = get_sentiment(company_profile)
        gemini_scam_likelihood = np.mean([desc_scam, req_scam, comp_scam])

        salary_range = str(job_data.get('salary_range', '') or '')
        unrealistic_salary = 0
        if salary_range and ('data entry' in title or 'no experience' in combined_text) and any(high in salary_range.lower() for high in ['$100,000', '$10,000 per month', 'huge income']):
            unrealistic_salary = 1

        features = {
            'desc_sentiment': desc_sent,
            'desc_subjectivity': desc_subj,
            'req_sentiment': req_sent,
            'req_subjectivity': req_subj,
            'company_sentiment': comp_sent,
            'company_subjectivity': comp_subj,
            'title_length': len(title),
            'description_length': len(description),
            'requirements_length': len(requirements),
            'title_word_count': len(title.split()),
            'description_word_count': len(description.split()),
            'missing_salary': 1 if not salary_range else 0,
            'missing_company': 1 if not company_profile else 0,
            'missing_requirements': 1 if not requirements else 0,
            'has_company_logo': 1 if job_data.get('has_company_logo') else 0,
            'has_questions': 1 if job_data.get('has_questions') else 0,
            'telecommuting': 1 if job_data.get('telecommuting') else 0,
            'scam_keyword_score': detect_scam_keywords(combined_text),
            'unrealistic_salary': unrealistic_salary,
            'gemini_scam_likelihood': gemini_scam_likelihood
        }
        features_list.append(features)
    
    return pd.DataFrame(features_list)

# Load and preprocess data
df = pd.read_csv('data/fake_job_postings.csv')  # Replace with your actual CSV path

# Check for 'fraudulent' column
if 'fraudulent' not in df.columns:
    raise ValueError("Missing 'fraudulent' column in CSV. Add it with 0/1 values for legitimate/fraudulent.")

numerical_features = extract_features(df)

# TF-IDF on combined text
combined_text = df['title'].fillna('') + ' ' + df['description'].fillna('') + ' ' + df['requirements'].fillna('') + ' ' + df['company_profile'].fillna('')
vectorizer = TfidfVectorizer(max_features=1000)
text_features = vectorizer.fit_transform(combined_text).toarray()

# Combine features
X = np.hstack([text_features, numerical_features.values])
y = df['fraudulent'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train with tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
clf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Evaluate
best_clf = grid_search.best_estimator_
preds = best_clf.predict(X_test)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Test accuracy: {accuracy_score(y_test, preds):.2%}")
print(classification_report(y_test, preds))

# Save PKL files
joblib.dump(best_clf, 'fraud_detection_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(numerical_features.columns.tolist(), 'feature_names.pkl')
