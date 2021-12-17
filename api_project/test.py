from api import app
import unittest
import pandas as pd
import requests

class FlaskTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.test_client(self)

    def test_data_load(self):
        payload = pd.read_csv('data/boston_train.csv', index_col=0).to_json()
        data = {'payload': payload, 'target_col_name': 'medv', 'columns_to_drop': None}
        response = self.client.post('http://127.0.0.1:8080/post_data', json=data)
        self.assertEqual(response.status_code, 200)

    def test_print_models(self):
        response = requests.get('http://127.0.0.1:8080/post_data')
        self.assertEqual(response.status_code, 200)

    def test_fit_model(self):
        model_params = {'fit_intercept': True, 'normalize': True}
        response = requests.post('http://127.0.0.1:8080/train_model/1', json=model_params)
        self.assertEqual(response.status_code, 200)

    def test_predict_model(self):
        payload = (pd.read_csv('data/boston_test.csv', index_col=0).to_json())
        response = requests.post('http://127.0.0.1:8080/predict/1', json=payload)
        self.assertEqual(response.status_code, 200)

    def test_alter_model(self):
        model_params = {'fit_intercept': True, 'normalize': True}
        payload = (pd.read_csv('data/boston_train.csv', index_col=0).to_json())
        data = {'payload': payload, 'target_col_name': 'medv', 'columns_to_drop': None, 'params': model_params}
        response = requests.put('http://127.0.0.1:8080/alter/1', json=data)
        self.assertEqual(response.status_code, 200)

    def test_delete_model(self):
        response = requests.delete('http://127.0.0.1:8080/alter/1')
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
