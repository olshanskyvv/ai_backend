import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from predictions.ML.data_reader import load_dataset


X_train_scaled, X_test_scaled, y_train, y_test = load_dataset()

# Шаг 4: Обучение модели нейронной сети
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Шаг 5: Прогноз и оценка модели нейронной сети
y_pred_nn = model.predict(X_test_scaled)

# Оценка модели
mse_nn = mean_squared_error(y_test, y_pred_nn)
mae_nn = mean_absolute_error(y_test, y_pred_nn)  # Вычисление MAE
r2_nn = r2_score(y_test, y_pred_nn)

print(f'Neural Network Model - Mean Squared Error: {mse_nn}')
print(f'Neural Network Model - Mean Absolute Error: {mae_nn}')  # Вывод MAE
print(f'Neural Network Model - R^2 Score: {r2_nn}')

# Визуализация результатов
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred_nn, label='Predicted (Neural Network)')
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