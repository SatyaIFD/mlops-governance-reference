from pydantic import BaseModel, Field, ConfigDict


class TransactionInputSchema(BaseModel):
    """
    Strict entry-point data schema checking payload consistency
    for raw transaction ingestion pipelines.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    Time: float = Field(
        ...,
        description="Seconds elapsed between this transaction and the first.",
        ge=0.0,
        allow_inf_nan=False,
    )
    Amount: float = Field(
        ...,
        description="Transactional monetary value magnitude.",
        ge=0.0,
        allow_inf_nan=False,
    )

    V1: float = Field(..., allow_inf_nan=False)
    V2: float = Field(..., allow_inf_nan=False)
    V3: float = Field(..., allow_inf_nan=False)
    V4: float = Field(..., allow_inf_nan=False)
    V5: float = Field(..., allow_inf_nan=False)
    V6: float = Field(..., allow_inf_nan=False)
    V7: float = Field(..., allow_inf_nan=False)
    V8: float = Field(..., allow_inf_nan=False)
    V9: float = Field(..., allow_inf_nan=False)
    V10: float = Field(..., allow_inf_nan=False)
    V11: float = Field(..., allow_inf_nan=False)
    V12: float = Field(..., allow_inf_nan=False)
    V13: float = Field(..., allow_inf_nan=False)
    V14: float = Field(..., allow_inf_nan=False)
    V15: float = Field(..., allow_inf_nan=False)
    V16: float = Field(..., allow_inf_nan=False)
    V17: float = Field(..., allow_inf_nan=False)
    V18: float = Field(..., allow_inf_nan=False)
    V19: float = Field(..., allow_inf_nan=False)
    V20: float = Field(..., allow_inf_nan=False)
    V21: float = Field(..., allow_inf_nan=False)
    V22: float = Field(..., allow_inf_nan=False)
    V23: float = Field(..., allow_inf_nan=False)
    V24: float = Field(..., allow_inf_nan=False)
    V25: float = Field(..., allow_inf_nan=False)
    V26: float = Field(..., allow_inf_nan=False)
    V27: float = Field(..., allow_inf_nan=False)
    V28: float = Field(..., allow_inf_nan=False)


class ProcessedTransactionSchema(BaseModel):
    """
    Downstream compliance data schema validating internal engine memory state
    post-feature transformation.
    Canonical column ordering: V1..V28, scaled_amount, scaled_time, Class.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    V1: float = Field(..., allow_inf_nan=False)
    V2: float = Field(..., allow_inf_nan=False)
    V3: float = Field(..., allow_inf_nan=False)
    V4: float = Field(..., allow_inf_nan=False)
    V5: float = Field(..., allow_inf_nan=False)
    V6: float = Field(..., allow_inf_nan=False)
    V7: float = Field(..., allow_inf_nan=False)
    V8: float = Field(..., allow_inf_nan=False)
    V9: float = Field(..., allow_inf_nan=False)
    V10: float = Field(..., allow_inf_nan=False)
    V11: float = Field(..., allow_inf_nan=False)
    V12: float = Field(..., allow_inf_nan=False)
    V13: float = Field(..., allow_inf_nan=False)
    V14: float = Field(..., allow_inf_nan=False)
    V15: float = Field(..., allow_inf_nan=False)
    V16: float = Field(..., allow_inf_nan=False)
    V17: float = Field(..., allow_inf_nan=False)
    V18: float = Field(..., allow_inf_nan=False)
    V19: float = Field(..., allow_inf_nan=False)
    V20: float = Field(..., allow_inf_nan=False)
    V21: float = Field(..., allow_inf_nan=False)
    V22: float = Field(..., allow_inf_nan=False)
    V23: float = Field(..., allow_inf_nan=False)
    V24: float = Field(..., allow_inf_nan=False)
    V25: float = Field(..., allow_inf_nan=False)
    V26: float = Field(..., allow_inf_nan=False)
    V27: float = Field(..., allow_inf_nan=False)
    V28: float = Field(..., allow_inf_nan=False)

    scaled_amount: float = Field(
        ...,
        description="RobustScaler transformed transaction dollar matrix weight.",
        allow_inf_nan=False,
    )
    scaled_time: float = Field(
        ...,
        description="StandardScaler transformed temporal shift weight.",
        allow_inf_nan=False,
    )
    Class: int = Field(
        ..., description="Target prediction binary state value.", ge=0, le=1
    )
