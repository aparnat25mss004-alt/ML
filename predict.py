
import joblib
import pandas as pd
from pathlib import Path

# Get project directory
BASE_DIR = Path(__file__).resolve().parent

# Load trained model
model_path = BASE_DIR / "models" / "student_models.pkl"

model = joblib.load(model_path)

print("Model loaded successfully!")

# Get new student details
new_student = pd.DataFrame({
    "study_hours": [7],
    "attendance": [85],
    "previous_marks": [70]
})

# Predict student result
prediction = model.predict(new_student)

# Display prediction
print("\nStudent Performance Prediction")

if prediction[0] == "Pass":
    print("Student Result: PASS")
else:
    print("Student Result: FAIL")

print("\nPrediction completed successfully!")
