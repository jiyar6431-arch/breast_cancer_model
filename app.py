import os
import joblib
import gradio as gr

# ==========================================================
# Load Model
# ==========================================================
try:
    deployed_dt = joblib.load("diabetes_prediction_model.pkl")
except Exception as e:
    print("Error loading model:", e)
    deployed_dt = None


# ==========================================================
# Prediction Function
# ==========================================================
def predict_diabetes(pregnancies, glucose, insulin, bmi, age):
    if deployed_dt is None:
        return "❌ Model could not be loaded."

    try:
        input_data = [[
            float(pregnancies),
            float(glucose),
            float(insulin),
            float(bmi),
            float(age)
        ]]

        prediction = deployed_dt.predict(input_data)

        if prediction[0] == 1:
            return "🔴 High Risk of Diabetes (Positive)"
        else:
            return "🟢 Low Risk of Diabetes (Negative)"

    except Exception as e:
        return f"Prediction Error: {e}"


# ==========================================================
# Developer Information
# ==========================================================
developer_info = """
## 👨‍💻 Created by Jiya Rana

**GitHub:** https://github.com/jiyar6431-arch

---

### 🛠️ Technologies Used

- Python
- Scikit-learn
- Decision Tree Classifier
- Gradio
- Render
"""


# ==========================================================
# Gradio Interface
# ==========================================================
interface = gr.Interface(
    fn=predict_diabetes,

    inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Glucose"),
        gr.Number(label="Insulin"),
        gr.Number(label="BMI"),
        gr.Number(label="Age"),
    ],

    outputs=gr.Textbox(label="Prediction Result"),

    title="🩺 Diabetes Prediction System",

    description="""
Enter the patient's medical information to predict the likelihood of diabetes
using a trained Decision Tree Machine Learning model.
""",

    article=developer_info,

    examples=[
        [2, 120, 80, 25.5, 35],
        [6, 170, 130, 35.2, 50],
        [0, 95, 60, 22.0, 24]
    ],

    theme=gr.themes.Soft()
)


# ==========================================================
# Launch App
# ==========================================================
if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
