"""
Ingestion Schema Definitions - Project 2: Loan Default Prediction
Defines the explicit data contract and exact categorical vocabularies for loan underwriting data.
"""

from typing import Any

RAW_LOAN_DATA_SCHEMA: dict[str, Any] = {
    "expected_columns": [
        "LoanID",
        "Age",
        "Income",
        "CreditScore",
        "LoanAmount",
        "DTIRatio",
        "MonthsEmployed",
        "NumCreditLines",
        "InterestRate",
        "LoanTerm",
        "Education",
        "EmploymentType",
        "MaritalStatus",
        "HasMortgage",
        "HasDependents",
        "LoanPurpose",
        "HasCoSigner",
        "Default",
    ],
    "numerical_types": {
        "Age": "int64",
        "Income": "int64",
        "CreditScore": "int64",
        "LoanAmount": "int64",
        "MonthsEmployed": "int64",
        "NumCreditLines": "int64",
        "LoanTerm": "int64",
        "Default": "int64",
        "DTIRatio": "float64",
        "InterestRate": "float64",
    },
    "categorical_columns": [
        "Education",
        "EmploymentType",
        "MaritalStatus",
        "HasMortgage",
        "HasDependents",
        "LoanPurpose",
        "HasCoSigner",
    ],
    "categorical_allowed_values": {
        "Education": ["Bachelor's", "High School", "Master's", "PhD"],
        "EmploymentType": ["Full-time", "Part-time", "Self-employed", "Unemployed"],
        "MaritalStatus": ["Divorced", "Married", "Single"],
        "HasMortgage": ["No", "Yes"],
        "HasDependents": ["No", "Yes"],
        "LoanPurpose": ["Auto", "Business", "Education", "Home", "Other"],
        "HasCoSigner": ["No", "Yes"],
    },
}
