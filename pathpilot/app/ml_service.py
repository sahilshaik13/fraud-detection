import joblib
import numpy as np
import pandas as pd
from textblob import TextBlob  # Fallback for sentiment
import re
import os
from django.conf import settings
import google.generativeai as genai  # For Gemini API
from dotenv import load_dotenv

load_dotenv()


class FraudDetectionService:
    def __init__(self, gemini_api_key=None):
        # Path to your trained models
        self.model_dir = os.path.join(settings.BASE_DIR, 'models')
        
        # Gemini API setup (requires API key)
        self.gemini_api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        self.gemini_model = None
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
            print("✅ Gemini API initialized for advanced analysis")
        else:
            print("⚠️ No Gemini API key provided. Falling back to TextBlob for sentiment.")
        
        # Load ML models
        model_path = os.path.join(self.model_dir, 'fraud_detection_model.pkl')
        vectorizer_path = os.path.join(self.model_dir, 'tfidf_vectorizer.pkl')
        features_path = os.path.join(self.model_dir, 'feature_names.pkl')
        
        self.model = None
        self.vectorizer = None
        self.feature_names = self._get_default_features()
        self.is_trained = False
        
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            try:
                self.model = joblib.load(model_path)
                self.vectorizer = joblib.load(vectorizer_path)
                if os.path.exists(features_path):
                    self.feature_names = joblib.load(features_path)
                self.is_trained = True
                print("✅ Loaded pre-trained fraud detection model")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
        
        if not self.is_trained:
            print("⚠️ No pre-trained model found. Using rule-based detection.")
    
    def _get_default_features(self):
        return [
            'desc_sentiment', 'desc_subjectivity', 'req_sentiment', 'req_subjectivity',
            'company_sentiment', 'company_subjectivity', 'title_length', 'description_length',
            'requirements_length', 'title_word_count', 'description_word_count',
            'missing_salary', 'missing_company', 'missing_requirements',
            'has_company_logo', 'has_questions', 'telecommuting', 'scam_keyword_score',
            'unrealistic_salary', 'gemini_scam_likelihood'
        ]
    
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text) or not text:
            return ""
        
        text = str(text)
        text = re.sub(r'#URL_[a-f0-9]+#', '', text)
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        text = re.sub(r'[^\w\s.,!?]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()
    
    def get_sentiment(self, text):
        """Extract sentiment polarity and subjectivity using Gemini API (fallback to TextBlob)"""
        cleaned = self.clean_text(text)
        if not cleaned:
            return 0.0, 0.0, 0.0  # Added scam_likelihood default
        
        if self.gemini_model:
            try:
                prompt = f"Analyze this job posting text for sentiment and scam indicators: '{cleaned}'. Return ONLY three floats separated by commas: polarity (-1 to 1), subjectivity (0 to 1), scam_likelihood (0 to 1)."
                print(f"📤 Sending prompt to Gemini: {prompt}")  # Debug print: Prompt sent
                response = self.gemini_model.generate_content(prompt)
                raw_response = response.text.strip()
                print(f"📥 Gemini raw response: {raw_response}")  # Debug print: Raw response
                parts = raw_response.split(',')
                if len(parts) == 3:
                    polarity, subjectivity, scam_likelihood = map(float, parts)
                    print(f"✅ Gemini parsed: Polarity={polarity}, Subjectivity={subjectivity}, Scam Likelihood={scam_likelihood}")  # Debug print: Parsed values
                    return polarity, subjectivity, scam_likelihood
                else:
                    raise ValueError("Invalid Gemini response format")
            except Exception as e:
                print(f"❌ Gemini sentiment error: {e}. Falling back to TextBlob.")
        
        # Fallback to TextBlob
        print("🔄 Falling back to TextBlob for sentiment analysis.")  # Debug print: Fallback
        blob = TextBlob(cleaned)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        scam_likelihood = 0.0  # Default for fallback
        print(f"✅ TextBlob parsed: Polarity={polarity}, Subjectivity={subjectivity}, Scam Likelihood={scam_likelihood} (default)")  # Debug print
        return polarity, subjectivity, scam_likelihood
    
    def detect_scam_keywords(self, text):
        """Detect common scam keywords and return a score (0-1)"""
        if not text:
            return 0.0
        
        scam_keywords = [
            'no experience needed', 'work from home', 'easy money', 'get paid weekly',
            'limited spots', 'apply now', 'free training', 'send money', 'bank details',
            'photo id', 'western union', 'paypal', 'huge income', 'earn up to', 'instant interview',
            'no skills required', 'quick cash', 'positions fill up quick', 'pay small fee', 'recruit friends',
            'act fast', 'spots left', 'from anywhere', 'independent work', 'starter kit'
        ]
        cleaned = self.clean_text(text).lower()
        matches = sum(1 for keyword in scam_keywords if keyword in cleaned)
        return min(matches / len(scam_keywords), 1.0)
    
    def extract_features(self, job_data):
        """Extract all features from job posting data"""
        if not isinstance(job_data, dict):
            raise ValueError("job_data must be a dictionary")
        
        # Clean texts
        title = self.clean_text(job_data.get('title', ''))
        description = self.clean_text(job_data.get('description', ''))
        requirements = self.clean_text(job_data.get('requirements', ''))
        company_profile = self.clean_text(job_data.get('company_profile', ''))
        
        # Combined text for keyword detection
        combined_text = ' '.join([title, description, requirements, company_profile])
        
        # Sentiment features (now returns scam_likelihood too)
        desc_sent, desc_subj, desc_scam = self.get_sentiment(description)
        req_sent, req_subj, req_scam = self.get_sentiment(requirements)
        comp_sent, comp_subj, comp_scam = self.get_sentiment(company_profile)
        
        # Average scam_likelihood from all texts
        gemini_scam_likelihood = np.mean([desc_scam, req_scam, comp_scam])
        
        # Unrealistic salary detection
        salary_range = job_data.get('salary_range', '')
        unrealistic_salary = 0
        if salary_range and ('data entry' in title.lower() or 'no experience' in combined_text) and any(high in salary_range.lower() for high in ['$100,000', '$10,000 per month', 'huge income']):
            unrealistic_salary = 1
        
        # Feature extraction
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
            'scam_keyword_score': self.detect_scam_keywords(combined_text),
            'unrealistic_salary': unrealistic_salary,
            'gemini_scam_likelihood': gemini_scam_likelihood  # New feature from Gemini
        }
        
        return features
    
    def rule_based_prediction(self, features):
        """Enhanced rule-based fraud detection with stricter rules"""
        fraud_score = 0.0
        reasons = []
        confidence = 0.7
        
        # Rule 1: Missing critical information
        if features['missing_company']:
            fraud_score += 0.30
            reasons.append("Missing company profile")
        if features['missing_salary']:
            fraud_score += 0.20
            reasons.append("Missing salary info")
        if features['missing_requirements']:
            fraud_score += 0.15
            reasons.append("Missing requirements")
        
        # Rule 2: Suspicious text characteristics
        if features['description_length'] < 50:
            fraud_score += 0.25
            reasons.append("Very short description")
        elif features['description_length'] < 100:
            fraud_score += 0.10
            reasons.append("Short description")
        if features['title_length'] < 5:
            fraud_score += 0.15
            reasons.append("Very short title")
        
        # Rule 3: Sentiment analysis (overly positive or subjective)
        if features['desc_sentiment'] > 0.5 or features['desc_sentiment'] < -0.1:
            fraud_score += 0.25
            reasons.append("Suspicious sentiment (overly positive/negative)")
        if features['desc_subjectivity'] > 0.6:
            fraud_score += 0.20
            reasons.append("High subjectivity")
        
        # Rule 4: Company credibility indicators
        if not features['has_company_logo']:
            fraud_score += 0.15
            reasons.append("No company logo")
        if not features['has_questions']:
            fraud_score += 0.10
            reasons.append("No screening questions")
        if features['telecommuting'] == 1:
            fraud_score += 0.20
            reasons.append("Telecommuting/remote job")
        
        # Rule 5: Scam keyword detection
        keyword_boost = features['scam_keyword_score'] * 0.50
        fraud_score += keyword_boost
        if keyword_boost > 0:
            reasons.append(f"Scam keywords detected (score: {features['scam_keyword_score']:.2f})")
        
        # Rule 6: Unrealistic salary
        if features['unrealistic_salary']:
            fraud_score += 0.25
            reasons.append("Unrealistic salary for job type")
        
        # Rule 7: Gemini scam likelihood (new)
        gemini_boost = features['gemini_scam_likelihood'] * 0.30
        fraud_score += gemini_boost
        if gemini_boost > 0:
            reasons.append(f"Gemini detected scam likelihood (score: {features['gemini_scam_likelihood']:.2f})")
        
        # Normalize
        fraud_probability = min(max(fraud_score, 0.0), 1.0)
        confidence = min(0.7 + (fraud_probability * 0.3), 1.0)
        
        return fraud_probability, confidence, reasons
    
    def ml_prediction(self, job_data):
        """Machine learning based prediction"""
        try:
            features = self.extract_features(job_data)
            combined_text = ' '.join([
                job_data.get('title', ''),
                job_data.get('description', ''),
                job_data.get('requirements', ''),
                job_data.get('company_profile', '')
            ])
            cleaned_text = self.clean_text(combined_text)
            text_features = self.vectorizer.transform([cleaned_text]).toarray()
            numerical_values = np.array([features[name] for name in self.feature_names]).reshape(1, -1)
            if text_features.shape[1] == 0 or numerical_values.shape[1] == 0:
                raise ValueError("Feature dimensions are empty")
            X = np.hstack([text_features, numerical_values])
            prediction = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            fraud_probability = probabilities[1]
            confidence = max(probabilities)
            return fraud_probability, confidence, []  # No reasons for ML
        except Exception as e:
            print(f"ML prediction error: {e}")
            return self.rule_based_prediction(self.extract_features(job_data))
    
    def predict_fraud(self, job_data):
        """Main prediction function"""
        try:
            if not isinstance(job_data, dict) or not job_data:
                raise ValueError("Invalid or empty job_data")
            features = self.extract_features(job_data)
            if self.is_trained and self.model and self.vectorizer:
                fraud_probability, confidence, reasons = self.ml_prediction(job_data)
                method = "ML Model"
            else:
                fraud_probability, confidence, reasons = self.rule_based_prediction(features)
                method = "Rule-based"
            is_fraudulent = fraud_probability > 0.5
            if fraud_probability >= 0.65:
                risk_level = 'High'
            elif fraud_probability >= 0.3:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
            result = {
                'is_fraudulent': is_fraudulent,
                'confidence': confidence,
                'fraud_probability': fraud_probability,
                'sentiment_score': features['desc_sentiment'],
                'risk_level': risk_level,
                'method': method,
                'reasons': reasons,
                'features': features
            }
            print(f"🔍 Fraud prediction completed using {method}")
            print(f"   Fraud Probability: {fraud_probability:.2%}")
            print(f"   Risk Level: {risk_level}")
            if reasons:
                print("   Reasons: " + "; ".join(reasons))
            return result
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {
                'is_fraudulent': True,
                'confidence': 0.5,
                'fraud_probability': 0.5,
                'sentiment_score': 0.0,
                'risk_level': 'Medium',
                'method': 'Error fallback',
                'reasons': [str(e)],
                'error': str(e)
            }
