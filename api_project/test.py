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
        response = self.client.post('/post_data', json=data)
        self.assertEqual(response.status_code, 200)

    def test_print_models(self):
        response = self.client.get('/post_data')
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()