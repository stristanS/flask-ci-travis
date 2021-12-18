import requests
import pandas as pd
import os

"""__________________ ТЕСТ НА ДАННЫХ BOSTON DATASET (РЕГРЕССИЯ)_______________________"""
# print(os.path.join(os.path.dirname(__file__)))

"""Загрузка данных для обучения"""
payload = pd.read_csv('data/boston_train.csv', index_col = 0).to_json()
data = {'payload':payload, 'target_col_name': 'medv', 'columns_to_drop': None}
response = requests.post('http://127.0.0.1:8080/post_data', json = data, timeout =5)
print(response, response.json())

"""Вывод моделей, доступных для обучения"""
response_1 = requests.get('http://127.0.0.1:8080/post_data')
print(response_1, response_1.json())

"""Обучение модели 1"""
model_params = {'fit_intercept': True, 'normalize': True}
response_2 = requests.post('http://127.0.0.1:8080/train_model/1', json = model_params)
print(response_2, response_2.json())
# #
# #
"""Предсказание модели 1"""
payload = (pd.read_csv('data/boston_test.csv', index_col = 0).to_json())
response_3 = requests.post('http://127.0.0.1:8080/predict/1', json =payload)
print(response_3, response_3.json())

"""Переобучение модели 1"""
payload = (pd.read_csv('data/boston_train.csv', index_col = 0).to_json())
data = {'payload':payload, 'target_col_name': 'medv', 'columns_to_drop': None, 'params': model_params}
response_4 = requests.put('http://127.0.0.1:8080/alter/1', json = data)
print(response_4, response_4.json())

"""Удаление модели 1"""
response_5 = requests.delete('http://127.0.0.1:8080/alter/1')
print(response_5, response_5.json())

# """__________________ ТЕСТ НА ДАННЫХ TITANIC DATASET _______________________"""
#
# """Загрузка данных для обучения"""
# payload = pd.read_csv('data/titanic_train.csv', index_col = 0).to_json()
# data = {'payload':payload, 'target_col_name': 'Survived', 'columns_to_drop': ['Name', 'Ticket']}
# response = requests.post('http://127.0.0.1:5000/post_data', json = data, timeout =5)
# print(response, response.json())
#
# """Вывод моделей, доступных для обучения"""
# response_1 = requests.get('http://127.0.0.1:5000/post_data')
# print(response_1, response_1.json())
#
# """Обучение модели 1"""
# model_params = {'fit_intercept': True}
# response_2 = requests.post('http://127.0.0.1:5000/train_model/2', json = model_params)
# print(response_2, response_2.json())
#
# """Предсказание модели 1"""
# payload = (pd.read_csv('data/titanic_test.csv', index_col = 0).to_json())
# response_3 = requests.post('http://127.0.0.1:5000/predict/2', json =payload)
# print(response_3, response_3.json())
#
# """Переобучение модели 1"""
# payload = (pd.read_csv('data/titanic_train.csv', index_col = 0).to_json())
# data = {'payload':payload, 'target_col_name': 'Survived', 'columns_to_drop': ['Name', 'Ticket'], 'params': model_params}
# response_4 = requests.put('http://127.0.0.1:5000/alter/2', json = data)
# print(response_4, response_4.json())
#
# """Удаление модели 1"""
# response_5 = requests.delete('http://127.0.0.1:5000/alter/2')
# print(response_5, response_5.json())
# #
# """__________________ ТЕСТ НА ДАННЫХ BREST CANCER DATASET _______________________"""
#
# """Загрузка данных для обучения"""
# payload = pd.read_csv('data/brest_canser_train.csv', index_col = 0).to_json()
# data = {'payload':payload, 'target_col_name': 'diagnosis', 'columns_to_drop': ['id']}
# response = requests.post('http://127.0.0.1:5000/post_data', json = data, timeout =5)
# print(response, response.json())
#
# """Вывод моделей, доступных для обучения"""
# response_1 = requests.get('http://127.0.0.1:5000/post_data')
# print(response_1, response_1.json())
#
# """Обучение модели 1"""
# model_params = {'fit_intercept': True}
# response_2 = requests.post('http://127.0.0.1:5000/train_model/2', json = model_params)
# print(response_2, response_2.json())
#
# """Предсказание модели 1"""
# payload = (pd.read_csv('data/brest_canser_test.csv', index_col = 0).to_json())
# response_3 = requests.post('http://127.0.0.1:5000/predict/2', json =payload)
# print(response_3, response_3.json())
#
# """Переобучение модели 1"""
# payload = (pd.read_csv('data/brest_canser_train.csv', index_col = 0).to_json())
# data = {'payload':payload, 'target_col_name': 'diagnosis', 'columns_to_drop': ['id'], 'params': model_params}
# response_4 = requests.put('http://127.0.0.1:5000/alter/2', json = data)
# print(response_4, response_4.json())
#
# """Удаление модели 1"""
# response_5 = requests.delete('http://127.0.0.1:5000/alter/2')
# print(response_5, response_5.json())