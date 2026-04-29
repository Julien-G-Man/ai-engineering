import uvicorn

from src.eta_predictor.main import app


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", 
                port=8000, reload=False
            )