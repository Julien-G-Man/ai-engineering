import requests, time
from time import sleep

helth_endpoint = "http://localhost:8000/health"
predict_endpoint = "http://localhost:8000/predict"

req = requests.get(helth_endpoint)
pred = requests.post(predict_endpoint)

for i in range(100):
    print(f"{i} Health status code: ", req.status_code)
    time.sleep(1)
