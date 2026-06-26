"""
Ingestion Schema Definitions - Project 2: Loan Default Prediction
Defines the explicit data contract for raw loan underwriting data.
"""

from typing import Dict, Any

RAW_LOAN_DATA_SCHEMA: Dict[str, Any] = {
    "expected_columns": [
        "LoanID", "Age", "Income", "CreditScore", "LoanAmount", "DTIRatio",
        "MonthsEmployed", "NumCreditLines", "InterestRate", "LoanTerm",
        "Education", "EmploymentType", "MaritalStatus", "HasMortgage",
        "HasDependents", "LoanPurpose", "HasCoSigner", "Default"
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
        "InterestRate": "float64"
    },
    "categorical_columns": [
        "Education", "EmploymentType", "MaritalStatus", 
        "HasMortgage", "HasDependents", "LoanPurpose", "HasCoSigner"
    ]
}
