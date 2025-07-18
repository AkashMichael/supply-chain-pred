import gradio as gr
import pandas as pd
from src.utils import load_object  # Adjust if utils path is different


def predict_demand(lag_1, lag_7, lag_30, rolling_mean_7, month, day_of_week):
    try:
        print("📥 Input Received:", lag_1, lag_7, lag_30, rolling_mean_7, month, day_of_week)

        # Load model and preprocessor
        model = load_object("artifacts/model.pkl")
        preprocessor = load_object("artifacts/preprocessor.pkl")
        print("✅ Model and preprocessor loaded.")

        # Create input DataFrame
        input_df = pd.DataFrame([{
            "lag_1": lag_1,
            "lag_7": lag_7,
            "lag_30": lag_30,
            "rolling_mean_7": rolling_mean_7,
            "month": month,
            "day_of_week": day_of_week
        }])
        print("📄 Input DataFrame:\n", input_df)

        # Optional debugging info
        print("🧪 Preprocessor expects columns:", preprocessor.get_feature_names_out() if hasattr(preprocessor, "get_feature_names_out") else "N/A")
        print("🧪 Input columns:", input_df.columns.tolist())

        # Transform and predict
        input_scaled = preprocessor.transform(input_df)
        prediction = model.predict(input_scaled)

        print("✅ Prediction:", prediction)
        return f"📦 Forecasted Demand: {prediction[0]:.2f} units"

    except Exception as e:
        print("❌ ERROR in prediction:", str(e))
        return f"❌ ERROR: {str(e)}"


# ✅ Build Gradio interface
interface = gr.Interface(
    fn=predict_demand,
    inputs=[
        gr.Number(label="Lag 1-Day Demand"),
        gr.Number(label="Lag 7-Day Demand"),
        gr.Number(label="Lag 30-Day Demand"),
        gr.Number(label="Rolling Mean (7 Days)"),
        gr.Slider(1, 12, step=1, label="Month"),
        gr.Slider(0, 6, step=1, label="Day of Week (0=Mon, 6=Sun)")
    ],
    outputs="text",
    title="📊 Product Demand Forecaster",
    description="Enter demand history and calendar info to forecast future demand."
)

# ✅ Launch the app
if __name__ == "__main__":
    interface.launch(share=True)
