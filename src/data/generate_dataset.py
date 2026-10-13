"""
Synthetic Customer Complaint Dataset Generator
Generates realistic customer complaints with diverse financial and service issues,
labels for category, sentiment, urgency, and recommended actions.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path for direct execution
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import random
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd

from src.config import (
    CATEGORIES,
    SENTIMENTS,
    URGENCIES,
    ACTIONS,
    RAW_DATA_PATH,
    SAMPLE_UPLOAD_PATH,
    DATA_DIR
)
from src.utils.logger import get_logger

logger = get_logger("dataset_generator")

# Seed for reproducible synthetic data
random.seed(42)

FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Sam", "Chris", "Pat", "Devon",
    "Casey", "Jamie", "Riley", "Cameron", "Avery", "Dakota", "Reese",
    "Peyton", "Skyler", "Kendall", "Hayden", "Logan", "Elena", "Marcus",
    "Priya", "Carlos", "Siddharth", "Fatima", "Chloe", "David", "Amina"
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"
]

TEMPLATES = {
    "Billing & Overdraft Fees": [
        "I was charged an unexpected overdraft fee of ${amount} on my checking account even though my direct deposit was posted the same morning. This is completely unfair and I demand a refund.",
        "Your system charged me three separate monthly maintenance fees totaling ${amount} without any warning. My balance was well above the required minimum. Please credit this back immediately.",
        "I noticed an unexpected monthly service fee of ${amount} on my statement from last Tuesday. I have never enrolled in paper statement delivery. Reverse this charge now.",
        "Why am I being assessed an international transaction fee of ${amount}? I made the purchase on a domestic US website. Kindly waive these duplicate charges.",
        "My account was debited twice for my internet utility payment, causing an NSF penalty fee of ${amount}. This was a technical glitch on your mobile app and I need a full fee waiver.",
        "You increased my interest rate and hit me with an unexplained late fee of ${amount}. I scheduled automatic payment 5 days before the due date! Refund my money.",
        "An unexpected bank charge and overdraft penalty of ${amount} occurred due to your system ledger processing lag. Please credit these fee deductions back.",
    ],
    "Account Access & Security": [
        "I have been locked out of my online banking account for over 48 hours. The SMS verification codes never arrive on my registered phone. I cannot access my funds to pay rent.",
        "Your biometric login failed repeatedly and now my profile is frozen. Customer support tells me to visit a branch, but there is no branch within 50 miles of my home.",
        "I reset my password three times, but the portal keeps throwing error 502 Bad Gateway. I urgently need to approve payroll today. Please unlock my credentials.",
        "Two-factor authentication prompt keeps rejecting my authenticator app tokens. I suspect a synchronization error on your security server. Please restore access.",
        "My mobile app crashed during facial authentication and locked my corporate account. The hotline put me on hold for 90 minutes. I cannot pay my vendors.",
        "Unable to sign into mobile banking due to continuous credential reset loop. Every time I input security questions it locks out my profile.",
    ],
    "Credit Reporting & Inquiries": [
        "Your bank erroneously reported a 60-day late payment to Experian, TransUnion, and Equifax for account #{acct}. My mortgage application was rejected because my credit score dropped by 75 points. Fix this reporting error immediately or I will contact the CFPB.",
        "I sent two dispute letters regarding a settled balance of ${amount} that is still marked as an active charge-off on my credit report. You have 30 days under the FCRA to delete this inaccurate trade line.",
        "A hard credit inquiry was conducted on my credit file without my written authorization on {date}. I never applied for your Platinum rewards card. Remove this inquiry right away.",
        "The loan payoff balance was satisfied 3 months ago, yet your automated reporting still shows an outstanding liability of ${amount}. Update my credit bureaus immediately.",
        "You reported my paid-in-full auto loan as delinquent to credit reporting agencies. My loan officer confirmed this error is on your end. Correct the bureau records today.",
        "Inaccurate negative reporting marks appear on TransUnion file claiming missed monthly installments. I have bank statements proving all payments cleared on time.",
    ],
    "Fraud & Unauthorized Transactions": [
        "URGENT: Someone stole ${amount} from my checking account via fraudulent debit card charges. I did not make these purchases and my card was stolen or cloned. Freeze account immediately!",
        "URGENT: I woke up to find 4 unauthorized wire transfers totaling ${amount} drained from my savings account to an unknown overseas account. I did NOT authorize this. Freeze my account and start a fraud investigation right now!",
        "Someone compromised my debit card ending in {last4} and spent ${amount} at an electronics retailer in another state. My card is still in my wallet. Block the card and issue a provisional credit.",
        "I received a text alert about an online purchase of ${amount} that I never made. I suspect my credentials were compromised in a phishing scam. Please stop payment immediately.",
        "Stolen identity alert: Someone opened a credit line in my name using my SSN 123-45-6789. I already filed a police report #{police_id}. Freeze all accounts associated with my identity.",
        "There is an unauthorized recurring charge of ${amount} billed every week from an unknown merchant named CryptoGlobal. I reported this fraud two weeks ago and no action was taken!",
        "Fraud alert: My debit card was charged ${amount} by an unauthorized third party scammer. Funds were drained without my consent. Reverse these fraudulent transactions.",
    ],
    "Loan & Mortgage Servicing": [
        "My monthly mortgage payment of ${amount} was sent via ACH on the 1st, but your servicing department has not applied it to the principal balance. You sent me a default warning letter by mistake.",
        "You calculated an escrow shortage of ${amount} that makes no sense. My property taxes actually decreased this year according to county records. Review my escrow analysis immediately.",
        "I applied for student loan income-driven repayment 90 days ago. The servicing status still says pending review, and interest of ${amount} continues to accrue unfairly.",
        "Your loan department lost my home equity modification documents for the third time. The foreclosure clock is ticking and my loan officer never answers emails.",
        "My auto loan payoff quote was ${amount}, which I wired on Monday. Now your agent claims I owe an extra processing fee before title release. Release my vehicle title immediately.",
    ],
    "Customer Service & Delays": [
        "I have called your helpline four times this week regarding my lost card. Each representative transferred me to another department and the call disconnected. Horrible customer service.",
        "I submitted a ticket to replace my damaged debit card two weeks ago. No confirmation email, no tracking number, and no card. Your chat agent was rude and unhelpful.",
        "Your customer representative hung up on me when I politely requested to speak with a supervisor regarding my disputed transaction. Completely unprofessional conduct.",
        "I have been waiting for a callback from a senior manager for 6 business days. Every time I call, I am promised an update within 24 hours that never happens.",
        "Your automated phone tree is a nightmare. It loops endlessly without giving an option to speak to a human representative. I spent 45 minutes trying to reach someone.",
    ],
}


def _generate_synthetic_sample(sample_id: int) -> Dict:
    category = random.choices(
        CATEGORIES,
        weights=[0.24, 0.18, 0.15, 0.20, 0.13, 0.10],
        k=1
    )[0]

    first_name = random.choice(FIRST_NAMES)
    last_name = random.choice(LAST_NAMES)
    customer_name = f"{first_name} {last_name}"
    customer_id = f"CUST-{random.randint(10000, 99999)}"
    tier = random.choices(["Standard", "Silver", "Gold", "VIP"], weights=[0.55, 0.25, 0.15, 0.05])[0]
    channel = random.choice(["Web Portal", "Mobile App", "Phone Support", "Secure Email"])

    template = random.choice(TEMPLATES[category])

    # Dynamic variables
    if category in ["Billing & Overdraft Fees", "Customer Service & Delays"]:
        amount = random.choice([25, 34, 35, 45, 70, 95, 120])
    elif category == "Fraud & Unauthorized Transactions":
        amount = random.choice([350, 780, 1250, 2400, 3850, 5000, 9200])
    elif category in ["Credit Reporting & Inquiries", "Loan & Mortgage Servicing"]:
        amount = random.choice([450, 890, 1800, 2300, 4500, 12000])
    else:
        amount = random.choice([50, 100, 250])

    last4 = str(random.randint(1000, 9999))
    acct = str(random.randint(10000000, 99999999))
    police_id = f"PR-{random.randint(10000, 99999)}"
    sample_date = (datetime.now() - timedelta(days=random.randint(1, 45))).strftime("%Y-%m-%d")

    text = template.format(
        amount=amount,
        last4=last4,
        acct=acct,
        police_id=police_id,
        date=sample_date
    )

    # Occasionally inject realistic raw PII into a subset of complaints to test PII masking agent
    if random.random() < 0.18:
        if category == "Fraud & Unauthorized Transactions":
            text += f" My SSN is {random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)} and my phone is (555) {random.randint(100,999)}-{random.randint(1000,9999)}."
        elif category == "Billing & Overdraft Fees":
            text += f" You can verify with my card 4532 8901 {random.randint(1000,9999)} {random.randint(1000,9999)}."

    # Determine ground-truth sentiment, urgency, and recommended action based on category and severity
    if category == "Fraud & Unauthorized Transactions":
        urgency = "Critical" if amount >= 1000 else "High"
        sentiment = "Extremely Frustrated / Hostile" if amount >= 1000 else "Dissatisfied"
        action = "Immediate Security Freeze & Fraud Review"
    elif category == "Credit Reporting & Inquiries":
        urgency = "High" if "mortgage" in text.lower() or "cfpb" in text.lower() else "Medium"
        sentiment = "Extremely Frustrated / Hostile" if "cfpb" in text.lower() else "Dissatisfied"
        action = "Credit Bureau Dispute Correction Workflow"
    elif category == "Loan & Mortgage Servicing":
        urgency = "Critical" if "foreclosure" in text.lower() or amount > 5000 else "High"
        sentiment = "Extremely Frustrated / Hostile" if urgency == "Critical" else "Dissatisfied"
        action = "Loan Servicing Specialist Outreach"
    elif category == "Billing & Overdraft Fees":
        urgency = "Medium" if amount > 50 else "Low"
        sentiment = "Dissatisfied" if amount > 50 else "Positive / Neutral"
        action = "Fee Waiver & Account Balance Adjustment"
    elif category == "Account Access & Security":
        urgency = "High" if "payroll" in text.lower() or "rent" in text.lower() else "Medium"
        sentiment = "Extremely Frustrated / Hostile" if urgency == "High" else "Dissatisfied"
        action = "Priority Tier 2 Callback & Investigation"
    else:  # Customer Service & Delays
        urgency = "Medium" if "hung up" in text.lower() else "Low"
        sentiment = "Extremely Frustrated / Hostile" if "hung up" in text.lower() else "Dissatisfied"
        action = "Standard Guidance & Self-Service FAQ" if urgency == "Low" else "Priority Tier 2 Callback & Investigation"

    # Timestamp within past 60 days
    days_ago = random.randint(0, 60)
    hours_ago = random.randint(0, 23)
    created_at = datetime.now() - timedelta(days=days_ago, hours=hours_ago)

    status_choices = ["New", "In Progress", "Under Review", "Resolved"]
    status_weights = [0.35, 0.25, 0.20, 0.20]
    status = random.choices(status_choices, weights=status_weights)[0]

    return {
        "complaint_id": f"CMP-{sample_id:04d}",
        "customer_id": customer_id,
        "customer_name": customer_name,
        "customer_tier": tier,
        "channel": channel,
        "complaint_text": text,
        "category": category,
        "sentiment": sentiment,
        "urgency": urgency,
        "recommended_action": action,
        "disputed_amount": float(amount),
        "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
    }


def generate_dataset(num_samples: int = 700, output_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Generates the full synthetic complaint dataset and saves it to CSV."""
    logger.info(f"Generating {num_samples} synthetic customer complaints...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = [_generate_synthetic_sample(i + 1000) for i in range(num_samples)]
    df = pd.DataFrame(samples)
    df.to_csv(output_path, index=False)
    logger.info(f"Dataset saved to {output_path} (Shape: {df.shape})")

    # Generate a lightweight sample batch CSV (without ground truth labels) for upload testing in dashboard
    sample_upload_df = df.sample(n=25, random_state=123)[
        ["complaint_id", "customer_id", "customer_name", "customer_tier", "channel", "complaint_text", "disputed_amount"]
    ].copy()
    SAMPLE_UPLOAD_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample_upload_df.to_csv(SAMPLE_UPLOAD_PATH, index=False)
    logger.info(f"Sample upload file saved to {SAMPLE_UPLOAD_PATH} (25 rows)")

    return df


if __name__ == "__main__":
    generate_dataset()
