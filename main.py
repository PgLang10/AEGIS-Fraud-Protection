from datetime import datetime
from typing import Optional
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import joblib
import numpy as np


# ---------------- LOAD MODEL ----------------

bundle = joblib.load("fraud_model.joblib")

model = bundle["model"]
FEATURES = bundle["features"]


# ---------------- CREATE APP ----------------

app = FastAPI(
    title="AEGIS Fraud Protection API"
)


# ---------------- IN-MEMORY DATABASE ----------------

TRANSACTIONS = []


# ---------------- TRANSACTION MODEL ----------------

class Transaction(BaseModel):

    customer_id: str
    payee: str

    new_payee: int

    amount: float
    typical_amount: float

    hour: int
    off_hours: int

    urgency_language_flag: int

    behavioral_anomaly_flag: int

    velocity_last_hour: int


# ---------------- GUARDIAN DECISION MODEL ----------------

class GuardianDecision(BaseModel):

    transaction_id: str

    approve: bool

    guardian_note: Optional[str] = None


# ---------------- RISK SCORING FUNCTION ----------------

def score_transaction(txn: Transaction):

    # Calculate amount relative to normal spending
    amount_ratio = (
        txn.amount /
        max(txn.typical_amount, 1.0)
    )


    # Feature dictionary
    feature_values = {

        "new_payee":
            txn.new_payee,

        "amount_ratio":
            amount_ratio,

        "off_hours":
            txn.off_hours,

        "urgency_language_flag":
            txn.urgency_language_flag,

        "behavioral_anomaly_flag":
            txn.behavioral_anomaly_flag,

        "velocity_last_hour":
            txn.velocity_last_hour
    }


    # Arrange features in training order
    x = np.array([
        [
            feature_values[feature]
            for feature in FEATURES
        ]
    ])


    # Get fraud probability
    risk_score = float(
        model.predict_proba(x)[0, 1]
    )


    # ---------------- EXPLAINABILITY ----------------

    coefficients = model.coef_[0]

    contributions = {

        feature:
        round(
            float(
                coefficient *
                feature_values[feature]
            ),
            3
        )

        for feature, coefficient
        in zip(
            FEATURES,
            coefficients
        )
    }


    # Sort highest contributors
    top_features = sorted(

        contributions.items(),

        key=lambda item:
        abs(item[1]),

        reverse=True

    )


    # ---------------- DECISION ----------------

    if risk_score < 0.30:

        decision = "allow"

    elif risk_score < 0.70:

        decision = (
            "step_up_verification"
        )

    else:

        decision = (
            "cooling_off_guardian_alert"
        )


    return {

        "risk_score":
            round(risk_score, 3),

        "decision":
            decision,

        "feature_contributions":
            contributions,

        "top_features":
            top_features[:3]
    }


# ==================================================
# API ENDPOINTS
# ==================================================


# ---------------- SCORE ONLY ----------------

@app.post("/score")

def score(txn: Transaction):

    return score_transaction(txn)


# ---------------- CREATE TRANSACTION ----------------

@app.post("/transactions")

def create_transaction(
    txn: Transaction
):

    result = score_transaction(txn)


    record = {

        "id":
            str(uuid.uuid4()),

        "timestamp":
            datetime.utcnow().isoformat(),

        "customer_id":
            txn.customer_id,

        "payee":
            txn.payee,

        "amount":
            txn.amount,

        **result,

        "status":

        (
            "pending_guardian"

            if result["decision"]
            ==
            "cooling_off_guardian_alert"

            else result["decision"]
        )
    }


    TRANSACTIONS.append(
        record
    )


    return record


# ---------------- AUDIT LOG ----------------

@app.get("/transactions")

def list_transactions():

    return list(
        reversed(
            TRANSACTIONS
        )
    )


# ---------------- GUARDIAN PENDING ----------------

@app.get("/guardian/pending")

def guardian_pending():

    return [

        transaction

        for transaction
        in TRANSACTIONS

        if transaction["status"]
        ==
        "pending_guardian"

    ]


# ---------------- GUARDIAN APPROVAL ----------------

@app.post("/guardian/approve")

def guardian_approve(
    decision: GuardianDecision
):

    for transaction in TRANSACTIONS:

        if (
            transaction["id"]
            ==
            decision.transaction_id
        ):

            transaction["status"] = (

                "approved"

                if decision.approve

                else "denied"

            )


            transaction[
                "guardian_note"
            ] = decision.guardian_note


            return transaction


    raise HTTPException(

        status_code=404,

        detail="Transaction not found"

    )


# ==================================================
# SERVE FRONTEND
# ==================================================

app.mount(

    "/",

    StaticFiles(
        directory="static",
        html=True
    ),

    name="static"

)