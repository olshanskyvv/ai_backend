import joblib
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from datetime import date

from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

xgb_model: XGBRegressor | None = None
scaler: StandardScaler | None = None

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

services = [
    'Аренда транспорта',
    'Услуги КР подвижного состава',
    'Услуги ТО подвижного состава',
    'Услуги ТР подвижного состава',
    'Услуги по управлению ПС',
]
transports = [
    'Дизельная тяга',
    'МВПС',
    'Рельсовые автобусы',
    'Скоростные поезда "Ласточка"',
    'Тепловозная тяга',
    'ЦМВ',
    'Электровозная тяга',
]
roads = [
    'Горьковская',
    'Куйбышевская',
    'Октябрьская',
    'Свердловская',
]

@app.get("/services")
async def get_services() -> dict[str, list[str]]:
    return {"services": services}


@app.get("/transports")
async def get_transports() -> dict[str, list[str]]:
    return {"transports": transports}


@app.get("/roads")
async def get_roads() -> dict[str, list[str]]:
    return {"roads": roads}


@app.get("/predict")
async def predict(service: str,
                  transport: str,
                  road: str,
                  date: date) -> dict[str, float]:
    if ((service not in services)
            or (transport not in transports)
            or (road not in roads)):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)

    date_data = {
        'Год': date.year, 'Месяц': date.month, 'День': 1,
    }
    ss = {'Услуга_' + s: service == s for s in services}
    print(ss)
    ts = {'Транспорт_' + t: transport == t for t in transports}
    print(ts)
    rs = {'Дорога_' + r: road == r for r in roads}
    print(rs)
    data = {**date_data, **rs, **ss, **ts}
    x = pd.DataFrame([data])
    x_scaled = scaler.transform(x)

    prediction = xgb_model.predict(x_scaled)
    return {'prediction': float(prediction[0])}


@app.on_event("startup")
async def startup():
    global xgb_model
    global scaler
    xgb_model = joblib.load('ML/xgb_model.pkl')
    scaler = joblib.load('ML/scaler.pkl')
