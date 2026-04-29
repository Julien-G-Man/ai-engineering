from fastapi import FastAPI

app = FastAPI(title="ETA Predictor", version="0.1.0")


@app.get("/")
def read_root():
    return {"message": "ETA Predictor API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}
