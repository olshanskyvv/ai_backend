import pandas as pd
import joblib

# Загрузка тестового файла
test_file_path = 'C:/Users/SAPIPA/Desktop/test_file.xlsx'
test_df = pd.read_excel(test_file_path)

# Преобразуем 'Дата' в формат datetime
test_df['Дата'] = pd.to_datetime(test_df['Дата'])
# Убедимся, что данные отсортированы по дате
test_df = test_df.sort_values('Дата')

# Выделим признаки и целевую переменную
X = test_df.drop(['Дата', 'Объем'], axis=1)

# Загрузка обученной модели
xgb_model = joblib.load('xgb_model.pkl')

# Преобразуем тестовые данные с помощью того же стандартизатора, что использовался для обучения модели
scaler = joblib.load('scaler.pkl')
X_test_scaled = scaler.transform(X)

# Предсказание
prediction = xgb_model.predict(X_test_scaled)

print("Предсказанное значение:", prediction)
