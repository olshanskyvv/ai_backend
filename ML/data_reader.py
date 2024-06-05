import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset(file_path: str = 'file.xlsx'):
    # Шаг 1: Загрузка данных
    df = pd.read_excel(file_path)

    # Шаг 2: Предварительный анализ данных
    print(df.head())
    print(df.info())
    print(df.describe())

    # Шаг 3: Подготовка данных

    # Преобразуем категориальные переменные в числовые
    df = pd.get_dummies(df, columns=['Дорога', 'Услуга', 'Транспорт'], drop_first=True)

    # Преобразуем 'Дата' в формат datetime
    df['Дата'] = pd.to_datetime(df['Дата'])

    # Убедимся, что данные отсортированы по дате
    df = df.sort_values('Дата')

    # Выделим признаки и целевую переменную
    X = df.drop(['Дата', 'Объем'], axis=1)
    y = df['Объем']

    # Разделим данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Стандартизация данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler