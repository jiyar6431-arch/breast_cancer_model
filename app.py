# app.py

import os
import gradio as gr
import joblib

# Load the trained Decision Tree model at startup
deployed_dt = joblib.load('diabetes_prediction_model.pkl')

def predict_diabetes(pregnancies, glucose, insulin, bmi, age):
    # The model expects a 2D array matching the exact order of x_train
    input_data = [[pregnancies, glucose, insulin, bmi, age]]
    prediction = deployed_dt.predict(input_data)
    
    # Interpret the binary outcome
    if prediction[0] == 1:
        return "Prediction: High Risk of Diabetes (Positive)"
    else:
        return "Prediction: Low Risk of Diabetes (Negative)"

# --- CODE BLOCK: ADDED FOOTER INFO AND ARTICLE PARAMETER ---
developer_info = """
### About the Developer
**Created by:** Jiya

* **LinkedIn:** [Connect with me](YOUR_LINKEDIN_URL_HERE)
* **GitHub:** [Check out my projects](https://github.com/jiyar6431-arch)
* **Instagram:** [Follow me](YOUR_INSTAGRAM_URL_HERE)

---
### 🛠️ Tools & Technologies Used
* **Machine Learning:** Scikit-learn (Decision Tree Classifier)
* **Web Framework:** Gradio
* **Language:** Python
* **Deployment:** Render
"""

interface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies (Number of times pregnant)"),
        gr.Number(label="Glucose (Plasma glucose concentration)"),
        gr.Number(label="Insulin (2-Hour serum insulin)"),
        gr.Number(label="BMI (Body mass index)"),
        gr.Number(label="Age (Years)")
    ],
    outputs=gr.Text(label="Assessment Result"),
    title="Diabetes Prediction System",
    description="Enter the medical metrics to predict diabetes risk using a Decision Tree Machine Learning model.",
    article=developer_info
)
# -----------------------------------------------------------

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
