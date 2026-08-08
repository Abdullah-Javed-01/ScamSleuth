from pathlib import Path

import pandas as pd


OUTPUT_PATH = Path(
    "data/stress_test/adversarial_stress_test.csv"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


examples = [
    # ==================================================
    # SAFE — deliberately contains words that could
    # otherwise look suspicious
    # ==================================================

    {
        "id": "ADV001",
        "label": "Safe",
        "scenario": "Employer-funded relocation",
        "text": (
            "The successful candidate may relocate to Dubai after probation. "
            "The company covers approved visa and travel costs directly. "
            "Applicants should never send money to an agent or recruiter."
        ),
    },

    {
        "id": "ADV002",
        "label": "Safe",
        "scenario": "Post-offer identity document",
        "text": (
            "Your signed employment agreement has been received. "
            "Please bring your original CNIC for verification on your first "
            "working day. Do not send identity documents through WhatsApp "
            "or email."
        ),
    },

    {
        "id": "ADV003",
        "label": "Safe",
        "scenario": "Legitimate training",
        "text": (
            "This paid internship includes two weeks of internal training. "
            "There is no training fee and candidates are not required to "
            "purchase any course or certification."
        ),
    },

    {
        "id": "ADV004",
        "label": "Safe",
        "scenario": "Normal online application",
        "text": (
            "Please submit your CV through our careers portal. "
            "The application form requests your name, email address, "
            "education, and work history. It does not request passwords, "
            "bank details, or payment-card information."
        ),
    },

    {
        "id": "ADV005",
        "label": "Safe",
        "scenario": "Normal compensation",
        "text": (
            "The Data Analyst role pays PKR 120,000 per month depending on "
            "experience. Shortlisted applicants complete a technical task "
            "and a live interview before a written offer is issued."
        ),
    },

    {
        "id": "ADV006",
        "label": "Safe",
        "scenario": "Company equipment",
        "text": (
            "Remote employees receive a company-owned laptop after joining. "
            "Candidates are not asked to buy equipment, send a deposit, "
            "or purchase devices from a selected vendor."
        ),
    },

    {
        "id": "ADV007",
        "label": "Safe",
        "scenario": "Finance role with money language",
        "text": (
            "The Accounts Assistant reconciles customer payments inside the "
            "company accounting system. Employees must never receive company "
            "funds into a personal bank account or forward client money."
        ),
    },

    {
        "id": "ADV008",
        "label": "Safe",
        "scenario": "Standard recruitment process",
        "text": (
            "Thank you for applying. The next stage is a thirty-minute video "
            "interview with the hiring manager, followed by reference checks "
            "and written employment terms if you are selected."
        ),
    },

    # ==================================================
    # SCAM — use wording different from the main
    # training-template language
    # ==================================================

    {
        "id": "ADV009",
        "label": "Scam",
        "scenario": "Disguised upfront fee",
        "text": (
            "Your place in the interview queue can be activated once the "
            "candidate administration charge of PKR 4,500 is transferred. "
            "The amount is required before scheduling."
        ),
    },

    {
        "id": "ADV010",
        "label": "Scam",
        "scenario": "Credential theft",
        "text": (
            "To connect your mailbox with the employee portal, reply with "
            "your email password and the one-time login code that appears "
            "on your phone."
        ),
    },

    {
        "id": "ADV011",
        "label": "Scam",
        "scenario": "Sensitive application portal",
        "text": (
            "Complete your candidate verification at "
            "https://verify-careers.example/profile. "
            "The form requires your debit-card number and CVV to confirm "
            "that the application belongs to you."
        ),
    },

    {
        "id": "ADV012",
        "label": "Scam",
        "scenario": "Selection bypass",
        "text": (
            "You have already been approved for the remote coordinator role. "
            "There will be no interview or screening stage. Sign the "
            "appointment form today so onboarding can begin."
        ),
    },

    {
        "id": "ADV013",
        "label": "Scam",
        "scenario": "Money mule",
        "text": (
            "During your trial week, supplier payments will arrive in your "
            "personal account. Keep ten percent as commission and forward "
            "the remaining funds to our settlement partner."
        ),
    },

    {
        "id": "ADV014",
        "label": "Scam",
        "scenario": "Cheque overpayment",
        "text": (
            "We will issue a cheque larger than your home-office expenses. "
            "Deposit it, use part of the amount for supplies, and send the "
            "unused balance to our logistics representative."
        ),
    },

    {
        "id": "ADV015",
        "label": "Scam",
        "scenario": "Mandatory paid certification",
        "text": (
            "Before your application can move forward, obtain the recruiter-"
            "approved compliance certificate. Enrollment costs USD 75 and "
            "must be completed through our training partner."
        ),
    },

    {
        "id": "ADV016",
        "label": "Scam",
        "scenario": "Applicant-funded equipment",
        "text": (
            "Your remote workstation must be ordered before your start date. "
            "Purchase the laptop from the supplier selected by HR and send "
            "the receipt so your employee account can be activated."
        ),
    },
]


df = pd.DataFrame(examples)

df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8",
)

print(
    "Stress-test dataset saved:",
    OUTPUT_PATH,
)

print(
    "Shape:",
    df.shape,
)

print(
    "\nLabel distribution:"
)

print(
    df["label"].value_counts()
)