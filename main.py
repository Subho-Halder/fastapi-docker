import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Load the saved models during application startup
pipeline_logistic = joblib.load('pipeline_logistic.pkl')
pipeline_random_forest = joblib.load('pipeline_random_forest.pkl')
pipeline_stacking = joblib.load('pipeline_stacking.pkl')

# Define a Pydantic model for input validation
class PredictionInput(BaseModel):
    Age: int
    Annual_Income: int
    Credit_Score: int
    Employment_Years: int
    Loan_Amount_Requested: int

# Define endpoint for home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})

# Define endpoint for prediction
@app.post("/predict", response_class=JSONResponse)
async def predict(request: Request, 
                  Age: int = Form(...), 
                  Annual_Income: int = Form(...), 
                  Credit_Score: int = Form(...), 
                  Employment_Years: int = Form(...), 
                  Loan_Amount_Requested: int = Form(...)):
    try:
        # Convert input data into a DataFrame
        data = {
            'Age': [Age],
            'Annual_Income': [Annual_Income],
            'Credit_Score': [Credit_Score],
            'Employment_Years': [Employment_Years],
            'Loan_Amount_Requested': [Loan_Amount_Requested]
        }
        df = pd.DataFrame(data)

        # Make predictions and probabilities
        predictions = {}

        # Logistic Regression
        logistic_pred = pipeline_logistic.predict(df)
        logistic_proba = pipeline_logistic.predict_proba(df)
        predictions['Logistic Regression'] = {
            'Loan_Default': int(logistic_pred[0]),
            'Probability_of_Default': float(logistic_proba[0][1])
        }

        # Random Forest
        rf_pred = pipeline_random_forest.predict(df)
        rf_proba = pipeline_random_forest.predict_proba(df)
        predictions['Random Forest'] = {
            'Loan_Default': int(rf_pred[0]),
            'Probability_of_Default': float(rf_proba[0][1])
        }

        # Stacking
        stacking_pred = pipeline_stacking.predict(df)
        stacking_proba = pipeline_stacking.predict_proba(df)
        predictions['Stacking'] = {
            'Loan_Default': int(stacking_pred[0]),
            'Probability_of_Default': float(stacking_proba[0][1])
        }

        return predictions

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI application with Uvicorn server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
