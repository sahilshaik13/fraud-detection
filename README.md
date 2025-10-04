# Job Posting Fraudulent Detection

An intelligent job posting analysis platform powered by AI to detect fraudulent listings.

***

## 📋 Overview

Job Posting Fraudulent Detection is a comprehensive machine learning project that analyzes and classifies job postings as real or fake. Leveraging NLP, structured feature extraction, and robust model training, this repository is designed to help job portals, recruiters, and job seekers identify scams and protect against fraudulent posts.

***

## ✨ Features

- 🧹 **Data Preprocessing** — Cleaning, imputation, and normalization of job posting data
- 🧬 **Feature Extraction** — NLP-based feature engineering from descriptions, requirements, and company profiles
- 🚦 **Fraud Classification** — Machine learning models (Random Forest, SVM, Logistic Regression) for fraud detection
- 🔄 **Continuous Learning** — Retraining pipelines for adaptive improvement
- 🖥️ **Web App** — Submit job postings for analysis, instant feedback
- 📦 **API Endpoints** — Fast, RESTful APIs for prediction and analytics
- 📊 **Analytics Dashboard** — Data visualizations for fraud rates, feature importance, and usage
- 👤 **User Management** — Profiles, histories, tracked predictions
- 🔒 **Security** — Secure handling of user data and Row Level Security in database

***

## 🛠️ Tech Stack

- **Backend:** Django web framework
- **WSGI Server:** Gunicorn
- **Machine Learning:** Scikit-learn, joblib, numpy, pandas
- **Database:** PostgreSQL, SQLAlchemy ORM
- **Database Adapter:** psycopg2-binary
- **Environment Management:** python-dotenv
- **NLP:** TextBlob
- **Authentication & Cloud:** google-auth, cloud-sql-python-connector
- **AI Services:** google-generativeai
- **Deployment:** Heroku / Render / local server

***

## 🧰 Core Dependencies

The backend and ML processing rely on:

| Package                      | Description                                  |
|------------------------------|----------------------------------------------|
| Django                       | Backend web framework                        |
| psycopg2-binary              | PostgreSQL database adapter for Python       |
| gunicorn                     | WSGI HTTP server for Django                  |
| python-dotenv                | Loads environment variables from `.env`      |
| numpy                        | Numerical computations                       |
| pandas                       | Data manipulation and analysis               |
| scikit-learn                 | Machine learning library                     |
| joblib                       | Model serialization and persistence          |
| textblob                     | NLP for sentiment and text analysis          |
| cloud-sql-python-connector   | Google Cloud SQL connections                 |
| SQLAlchemy                   | Database ORM and toolkit                     |
| google-auth                  | Auth for Google Cloud APIs                   |
| google-generativeai          | Access Google generative AI models           |

***

## 📋 Environment Variables & API Keys

Add the following variables to your `.env` file to securely connect to databases and APIs:

| Variable Name                        | Description                                   | Required |
|--------------------------------------|-----------------------------------------------|----------|
| DATABASE_URL                         | PostgreSQL DB connection string               | ✅       |
| GOOGLE_APPLICATION_CREDENTIALS       | Path to Google service account file           | ✅       |
| GENERATIVE_AI_API_KEY                | Key for google-generativeai                   | ✅       |
| SQLALCHEMY_DATABASE_URI              | SQLAlchemy DB URI                             | ✅       |
| CLOUD_SQL_CONNECTION_NAME            | Google Cloud SQL instance ID                  | ✅       |
| SECRET_KEY                           | Django secret key                             | ✅       |
| APP_URL                              | App base URL                                  | ✅       |

### Example `.env`

```dotenv
# Django
SECRET_KEY=your_django_secret_key

# Database
DATABASE_URL=postgres://user:password@host:port/db_name

# Google Services
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
GENERATIVE_AI_API_KEY=your_google_genai_api_key
CLOUD_SQL_CONNECTION_NAME=your_cloud_sql_instance

# ORM
SQLALCHEMY_DATABASE_URI=postgresql+psycopg2://user:password@host:port/db_name

APP_URL=http://localhost:8000
```

> Never commit sensitive API keys or `.env` files to your public repository—use `.gitignore` to exclude `.env` and share instructions in documentation.

***

## 🚀 Installation

**Clone the repository**
```bash
git clone https://github.com/Shaik-Farhana/job-posting-fraudulent-detection.git
cd job-posting-fraudulent-detection
```

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Set up environment variables**
Copy example file and add your credentials:
```bash
cp .env.example .env
```

**Set up database and Cloud/Google services**
Run migrations or initialize tables, configure Google credentials.

**Start the app**
```bash
gunicorn app.wsgi        # Django + Gunicorn
# Or for development
python manage.py runserver
```
Visit http://localhost:8000

***

## 📁 Project Structure

```
├── data/                     # Datasets
├── notebooks/                # Jupyter analysis notebooks
├── models/                   # Saved ML models
├── app/                      # Django project and apps
│   ├── static/               # Static assets
│   ├── templates/            # HTML templates
├── api/                      # API endpoints
├── dashboard/                # Analytics views
├── utils/                    # Utility scripts
├── scripts/                  # Automated scripts
├── .env                      # Environment variables
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
└── LICENSE                   # License
```

***

## 🗄️ Database Schema

### Core Tables

| Table                   | Description                                                             |
|-------------------------|-------------------------------------------------------------------------|
| user_profiles           | User info, emails, login data                                           |
| job_postings            | Core posting data; title, description, meta-info                        |
| ratings                 | User ratings for posted jobs (optional)                                 |
| predictions             | Results of fraud analysis predictions                                   |
| user_history            | User prediction and posting history                                     |
| companies               | Company profile data                                                    |

### Key Design

- **UUID Primary Keys** for user/posting entries
- **Array Fields** for skills, requirements, languages, etc.
- **JSONB** for flexible storage (company profile/details)
- **Row Level Security** for access protection (Supabase/Postgres)

***

## 🔌 API Routes

### Authentication

- `POST /api/auth/register` — Register a new user
- `POST /api/auth/login` — Login

### Job Post Management

- `GET /api/jobs` — List job postings with filters
- `GET /api/jobs/{id}` — Detail for a specific posting
- `POST /api/jobs/predict` — Submit job posting for fraud prediction
- `POST /api/jobs/{id}/rating` — Add rating
- `POST /api/jobs/{id}/review` — Add review

### Analytics

- `GET /api/analytics/fraud-rate` — % of fraudulent postings detected
- `GET /api/analytics/feature-importance` — Visual importance of model features

### User Management

- `GET /api/user/profile`
- `PUT /api/user/profile`

***

## 🎯 User Journey

1. **Registration & Onboarding**
   - User signs up with email/password
   - Sets profile preferences and privacy options
   - Can mark past postings analyzed or watched

2. **Job Discovery & Analysis**
   - Submit new job posting OR browse loaded postings dataset
   - See prediction, detailed fraud analysis, model explanation

3. **Community Interactions**
   - Rate postings, review suspicious jobs, help train models

4. **Model Improvement & Feedback**
   - Retrain models with new data
   - Dashboard visualizes overall fraud trends, user accuracy statistics

***

## 🤖 AI Fraud Detection System

**How It Works:**
- User/job data is sent to the backend, which uses Gemini 1.5 Flash (Google's Gemini LLM) for prediction:
   ```
   {
      "title": "Marketing Intern",
      "description": "...",
      "company_profile": "...",
      "requirements": "...",
      "location": "Remote",
      "employment_type": "Full-time"
   }
   ```
- The Gemini model predicts `is_fraud`: 0 (real) or 1 (fraudulent)
- Results are cached in the predictions table
- Dashboard displays summary analytics, progress, and feature impacts

**Smart Caching Strategy**
- Prediction results are stored and the cache is refreshed if posting or user data changes
- Reduces compute cost and speeds up access

**Model Details**
- Gemini 1.5 Flash is used as the core LLM for fraud detection
- The backend integrates with Google Generative AI APIs to access Gemini models for real-time predictions
- Model outputs are combined with structured features for improved accuracy

***
## 📊 Analytics Features

Dashboard provides insights including:
- **Fraud Rate Trends** — Proportion of fake job postings found
- **Feature Importance** — Key attributes associated with fraud
- **User Activity Patterns** — Track predictions, ratings, reviews

***

## 🎨 UI Components

- **JobCard** — Summary display for each job
- **ProfileForm** — User profile management
- **PredictButton** — Run fraud detection
- **Rating** — Interactive user ratings
- **ReviewDialog** — Modal for user feedback
- **Sidebar** — Main navigation
- (If React frontend) Styled using Tailwind CSS and shadcn/ui

***

## 🔒 Security Features

- **Row Level Security** enabled for all user and job data tables
- **Environment Variables** for sensitive credentials
- **API Auth Protection** on endpoints
- **Type Validation** and input sanitization
- **TypeScript** on frontend for added safety

***

## 🚀 Deployment

**Development**
```bash
python app.py           # Flask/Django server
npm run dev             # React frontend
```
**Production**
- Deploy backend (Heroku, Render, Vercel)
- Connect Supabase/Postgres for database & auth
- Set environment variables in dashboard

**Supabase Setup**
- Create project, import schema/tables
- Enable Row Level Security
- Configure authentication

***

## 🧪 Testing

- `pytest` for backend tests
- `npm run test` for frontend
- `npm run test:coverage` to generate test reports

***

## ⚡ Performance Features

- Caching predictions for repeat requests
- Database indexing on queried fields (job_id, user_id)
- Static generation of analytics pages where possible
- Image optimization for company logos

***

## 🔧 Configuration Files

- `.env` — Environment variables
- `next.config.js` — Next.js (if React frontend)
- `tsconfig.json` — TypeScript definitions
- `postcss.config.js` — CSS setup for Tailwind

***

## 📱 Mobile Support

- Responsive design with Tailwind CSS
- Touch-friendly UI for browsing, rating, review submission

***

## 🔄 State Management

- Python app: FastAPI/Flask session, database sync
- Frontend: React Context, hooks for logic and local state

***

## 🤝 Contributing

1. Fork repository
2. `git checkout -b feature/amazing-feature`
3. Commit (`git commit -m 'Add amazing feature'`)
4. Push (`git push origin feature/amazing-feature`)
5. Open Pull Request

Development Guidelines:
- Follow Python, React & TypeScript best practices
- Add/review tests for new functionality
- Document API endpoints and models
- Use conventional commits

***

## 📋 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| DATABASE_URL | PostgreSQL DB connection | ✅ |
| SUPABASE_URL | Supabase project URL | ✅ |
| SUPABASE_ANON_KEY | Supabase anon key | ✅ |
| SUPABASE_SERVICE_ROLE_KEY | Supabase service role | ✅ |
| APP_URL | App base URL | ✅ |

***

## 🐛 Troubleshooting

### Common Issues

- **Build Errors**: Check for Python/TypeScript errors, missing modules
- **Database**: Validate connection string, permissions, schema
- **Prediction Issues**: Check model path, dependencies, input format

***

## 🛣️ Roadmap

**Phase 1 (Current)**
- Core fraud detection models and web interface
- User registration and onboarding
- Feature analytics dashboard

**Phase 2 (Upcoming)**
- Social user features (upvotes, sharing)
- Watchlist and favorites
- Advanced data filtering

**Phase 3 (Future)**
- Enhanced ML models
- Real-time job data scraping
- External API integration (LinkedIn, Indeed)
- Advanced analytics

***

## 📄 License

This project is licensed under the MIT License — see the LICENSE file.

***

## 🙏 Acknowledgments

- Scikit-learn, Pandas — ML and data analysis
- Supabase, PostgreSQL — Backend and authentication
- Flask/Django — Web server
- React, Tailwind CSS, shadcn/ui — Frontend/UI (if used)
- Heroku, Vercel, Render — Deployment platforms
- All contributors and the open-source community

⭐ Star this repository if you find it helpful!

***

Live Demo: https://fraud-detect-service-810180013725.asia-south1.run.app/
