from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from predict_and_prepare import predict_and_prepare

app = FastAPI()

@app.get("/predict/")
def get_icu_risk(source: str = Query(...), patient_id: int = Query(...)):
    # Determine file path based on source
    if source == "original":
        file_path = "Kaggle_Sirio_Libanes_ICU_Prediction.xlsx"
    elif source == "synthetic":
        file_path = "synthetic_data.xlsx"
    else:
        return JSONResponse(status_code=400, content={"error": "Invalid source"})

    model_path = "cnn_icu_model.h5"

    try:
        df = predict_and_prepare(file_path, model_path)
        patient_data = df[df['PATIENT_VISIT_IDENTIFIER'] == patient_id]
        if patient_data.empty:
            return JSONResponse(status_code=404, content={"error": "Patient ID not found"})
        return patient_data.to_dict(orient="records")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
