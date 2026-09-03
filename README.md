AEGIS – AI-Powered Fraud Protection System
Problem Statement

Digital financial fraud disproportionately affects vulnerable customers, including senior citizens, first-time digital banking users, and digitally inexperienced individuals.

AEGIS is an intelligent fraud protection system designed to proactively identify suspicious transactions and provide additional protection before money is lost.

Features
     -AI-powered transaction risk scoring
     -Logistic Regression fraud detection model
     -Explainable risk predictions
     -Per-feature contribution analysis
     -New payee detection
     -Unusual transaction amount detection
     -Behavioral anomaly detection
     -Transaction velocity monitoring
     -Urgency language detection
     -Step-up verification for suspicious transactions
     -Cooling-off period for high-risk transactions
     -Guardian approval system
     -Transaction audit log

How It Works

AEGIS analyzes multiple transaction risk factors:

     1. New or unfamiliar payee
     2. Transaction amount compared with the customer's typical spending
     3. Time of transaction
     4. Urgency or social-engineering indicators
     5. Behavioral anomalies
     6. High transaction velocity

The Logistic Regression model generates a fraud probability.

Based on the risk score:

     -Low Risk → Transaction allowed instantly
     -Medium Risk → Step-up verification required
     -High Risk → Cooling-off period and guardian alert
     
Explainable AI

AEGIS uses Logistic Regression instead of a black-box-only approach.

Each feature has a model coefficient, allowing the system to explain why a transaction received a particular risk score.

Feature contribution:

contribution = coefficient × feature value

This allows the Risk Engine Log to display the factors that contributed most to a transaction's risk score.

Technology Stack
     -Python
     -FastAPI
     -Scikit-learn
     -Logistic Regression
     -Pandas
     -NumPy
     -Joblib
     -HTML, CSS and JavaScript

Project Structure

AEGIS-Fraud-Protection/
│
├── main.py
├── generate_dataset.py
├── train_model.py
├── fraud_model.joblib
├── transactions.csv
├── requirements.txt
├── .gitignore
└── index.html
Installation

Clone the repository:

git clone https://github.com/YOUR-USERNAME/AEGIS-Fraud-Protection.git

Install dependencies:
pip install -r requirements.txt

Generate Dataset
python generate_dataset.py

Train the Model
python train_model.py

Run the Application
uvicorn main:app --reload

Open the application in your browser:

http://127.0.0.1:8000

API Endpoints
Endpoint	                    Description
POST /score	               Calculates transaction fraud risk
POST /transactions	          Creates and stores a transaction
GET /transactions	          Returns transaction audit logs
GET /guardian/pending	     Returns transactions awaiting guardian approval
POST /guardian/approve	     Guardian approves or denies a transaction

AEGIS Protection Flow

Transaction
     ↓
Risk Feature Analysis
     ↓
Logistic Regression Model
     ↓
Risk Score + Explainability
     ↓
Low Risk → Allow
Medium Risk → Step-up Verification
High Risk → Cooling-off + Guardian Alert

Hackathon Project

AEGIS was developed as a prototype for protecting vulnerable customers from digital financial fraud and scam-based transactions.

The system focuses on proactive prevention, explainable AI, and human-in-the-loop protection.
