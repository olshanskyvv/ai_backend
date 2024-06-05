import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

from ML.data_reader import load_dataset


X_train_scaled, X_test_scaled, y_train, y_test, scaler = load_dataset()

# Шаг 4: Обучение модели градиентного бустинга
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# Шаг 5: Прогноз и оценка модели градиентного бустинга
y_pred_xgb = xgb_model.predict(X_test_scaled)

joblib.dump(xgb_model, 'xgb_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Оценка модели
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)  # Вычисление MAE
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f'XGBoost Model - Mean Squared Error: {mse_xgb}')
print(f'XGBoost Model - Mean Absolute Error: {mae_xgb}')  # Вывод MAE
print(f'XGBoost Model - R^2 Score: {r2_xgb}')

# Визуализация результатов
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred_xgb, label='Predicted (XGBoost)')
plt.legend()
plt.show()


#Этот код выполняет следующие действия:
#Загружает данные из файла Excel.
#Проводит предварительный анализ данных, выводя первые несколько строк, информацию о данных и статистику.
#Преобразует столбец даты в формат datetime и сортирует данные по дате.
#Выделяет признаки и целевую переменную.
#Разделяет данные на обучающую и тестовую выборки и стандартизирует их.
#Обучает модель случайного леса на обучающих данных.
#Делает прогнозы и оценивает модель, выводя метрики MSE и R^2, а также строит график для визуализации результатов.



#Мне нужно в рамках выпускной квалификационной работы по искусственному интеллекту
#оптимизировать какую-либо работу в РЖД по средствам внедрения машинного обучения.
#Я хочу, чтобы ИИ мог предсказывать объем оказываемых услуг на следующий месяц. 
#Напиши пожалуйста код для этого используя дата сет, который я тебе прикрепил   