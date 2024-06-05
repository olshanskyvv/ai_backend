import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression

from data_reader import load_dataset


X_train_scaled, X_test_scaled, y_train, y_test = load_dataset()


# Шаг 4: Обучение модели линейной регрессии
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Шаг 5: Прогноз и оценка модели
y_pred_linear = linear_model.predict(X_test_scaled)

# Оценка модели
mse_linear = mean_squared_error(y_test, y_pred_linear)
mae_linear = mean_absolute_error(y_test, y_pred_linear)  # Вычисление MAE
r2_linear = r2_score(y_test, y_pred_linear)

print(f'Linear Model - Mean Squared Error: {mse_linear}')
print(f'Linear Model - Mean Absolute Error: {mae_linear}')  # Вывод MAE
print(f'Linear Model - R^2 Score: {r2_linear}')

# Визуализация результатов
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred_linear, label='Predicted (Linear Regression)')
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