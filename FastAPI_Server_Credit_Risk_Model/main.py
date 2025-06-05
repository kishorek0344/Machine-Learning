from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from prediction_helper import predict

class CreditRiskInput(BaseModel):
    age: int
    income: float
    loan_amount: float
    loan_tenure_months: int
    avg_dpd_per_deliquency: float
    deliquency_ratio: float
    credit_utilization_ratio: float
    num_open_accounts: int
    residence_type: str
    loan_purpose: str
    loan_type: str

class CreditRiskOutput(BaseModel):
    probability: float
    credit_score: int
    rating: str


app = FastAPI()

@app.get("/ping")
def ping():
    return "Hello, i am the server"

#predict_credit_risk
@app.post("/predict_credit_risk", response_model=CreditRiskOutput)
def predict_credit_risk(input_data: CreditRiskInput):
    try:
        probability, credit_score, rating = predict(input_data.age, input_data.income, input_data.loan_amount,
                                                    input_data.loan_tenure_months, input_data.avg_dpd_per_deliquency,
                                                    input_data.deliquency_ratio, input_data.credit_utilization_ratio,
                                                    input_data.num_open_accounts, input_data.residence_type,
                                                    input_data.loan_purpose, input_data.loan_type)
        return CreditRiskOutput(probability=probability, credit_score=credit_score, rating=rating)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))