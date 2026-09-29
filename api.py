import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates

# create FastAPI Object
app = FastAPI(title="Student Placement Prediction API")

# Load the trained model once at startup
model = pickle.load(open("placement_model.pkl", "rb"))

templates = Jinja2Templates(directory="templates")

# Input format with validation (FastAPI rejects values outside these ranges)
class Student(BaseModel):
    IQ: int = Field(..., ge=50, le=200, example=110)
    CGPA: float = Field(..., ge=0, le=10, example=8.2)
    tenth_marks: float = Field(..., ge=0, le=100, example=85)
    twelfth_marks: float = Field(..., ge=0, le=100, example=80)
    Communication_Skills: int = Field(..., ge=1, le=10, example=8)

# @app.get("/")
# def home():
#     return {"message": "Welcome to Placement Prediction API is running. Visit /docs to try it."}

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.post("/predict")
def predict(student: Student):
    # Column names must match the ones used during training
    data = pd.DataFrame([{
        "IQ": student.IQ,
        "CGPA": student.CGPA,
        "10th_Marks": student.tenth_marks,
        "12th_Marks": student.twelfth_marks,
        "Communication_Skills": student.Communication_Skills,
    }])

    prediction = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0][1])

    return {
        "placed": prediction,
        "result": "Placed" if prediction == 1 else "Not Placed",
        "probability": round(probability, 2),
    }
