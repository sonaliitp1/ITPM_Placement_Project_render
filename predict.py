import pickle
import pandas as pd

model = pickle.load(open("placement_model.pkl", "rb"))

# Change these values to test different students
student = pd.DataFrame([{
    "IQ": 110,
    "CGPA": 8.2,
    "10th_Marks": 85,
    "12th_Marks": 80,
    "Communication_Skills": 8,
}])

prediction = model.predict(student)[0]
probability = model.predict_proba(student)[0][1]   # probability of being placed

print("Student:", student.iloc[0].to_dict())
print("Prediction:", "PLACED" if prediction == 1 else "NOT PLACED")
print(f"Placement probability: {probability:.0%}")
