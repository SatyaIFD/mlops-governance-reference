from pydantic import BaseModel, Field, ConfigDict

class TransactionInputSchema(BaseModel):
    """
    Strict entry-point data schema checking payload consistency
    for raw transaction ingestion pipelines.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    Time: float = Field(..., description="Seconds elapsed between this transaction and the first.", ge=0.0)
    Amount: float = Field(..., description="Transactional monetary value magnitude.", ge=0.0)
    
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float; V8: float
    V9: float; V10: float; V11: float; V12: float; V13: float; V14: float; V15: float; V16: float
    V17: float; V18: float; V19: float; V20: float; V21: float; V22: float; V23: float; V24: float
    V25: float; V26: float; V27: float; V28: float

class ProcessedTransactionSchema(BaseModel):
    """
    Downstream compliance data schema validating internal engine memory state
    post-feature transformation.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    scaled_amount: float = Field(..., description="RobustScaler transformed transaction dollar matrix weight.")
    scaled_time: float = Field(..., description="StandardScaler transformed temporal shift weight.")
    
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float; V8: float
    V9: float; V10: float; V11: float; V12: float; V13: float; V14: float; V15: float; V16: float
    V17: float; V18: float; V19: float; V20: float; V21: float; V22: float; V23: float; V24: float
    V25: float; V26: float; V27: float; V28: float
    
    Class: int = Field(..., description="Target prediction binary state value.", ge=0, le=1)