from flask import Flask
from flask_restx import Api, Resource
from flask import request, jsonify
import os
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import pandas as pd
import pickle
from sklearn import preprocessing
import logging
from prometheus_flask_exporter import PrometheusMetrics
import mlflow.sklearn

logging.basicConfig(filename='../storage/record.log', level=logging.DEBUG, format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s %(funcName)s: %(message)s')
app = Flask(__name__)
api = Api(app)
metrics = PrometheusMetrics(app, default_latency_as_histogram=False)
mlflow.set_tracking_uri('http://mlflow:5001')


class MLModelsDAO:
    def __init__(self):
        self.model_file_name = '../storage/'
        self.data = None
        self.target_col_name = None
        self.columns_to_drop = None
        self.train_x_ohe = None
        self.models = [{'model_id': 1, 'model': LinearRegression()}, {'model_id': 2, 'model': LogisticRegression(solver='liblinear')}]

    @metrics.counter('my_custom_count_get_models_for_ml', 'Number of requests', labels={'status': lambda resp: resp.status_code})
    def get(self):
        models_dict = {}
        for model in self.models:
            models_dict[model['model_id']] = str(model['model'])
        return models_dict

    def post(self, data):
        try:
            logging.info('Posting data in progress...')
            self.data = pd.read_json(data['payload'])
            self.target_col_name = data['target_col_name']
            self.columns_to_drop = data['columns_to_drop']
            if self.target_col_name not in self.data.columns:
                logging.error('No such column {} in dataframe.'.format(self.target_col_name))
                api.abort(400, 'No such column {} in dataframe.'.format(self.target_col_name))
            if self.columns_to_drop is not None:
                for col in self.columns_to_drop:
                    if col not in self.data:
                        logging.error('No such column {} in dataframe.'.format(col))
                        api.abort(400, 'No such column {} in dataframe.'.format(col))
            logging.info('Posting data successfully finished')
            return {'status_code': 200}
        except KeyError:
            logging.error('Invalid data format')
            api.abort(400, 'Input format allowed: {payload: .json, target_col_name: str, columns_to_drop: list/None')
        except MemoryError:
            logging.error('MemoryError')
            api.abort(400, 'MemoryError, unable to allocate data')

    def data_preprocessing(self, data):
        try:
            logging.info('Preprocessing data in progress...')
            if self.columns_to_drop is not None:
                data.drop(self.columns_to_drop, axis=1, inplace=True)
            x, y = data.loc[:, data.columns != self.target_col_name], data[self.target_col_name]
            cat_features = []
            for col in x.columns:
                if x[col].dtypes == 'O':
                    cat_features.append(col)
                    x[col].fillna('unseen_cat', inplace=True)
                else:
                    x[col].fillna(-1, inplace=True)
            x = pd.get_dummies(x, columns=cat_features)
            if y.dtypes == 'O':
                le = preprocessing.LabelEncoder()
                y = le.fit_transform(y)
            # x, y = data.loc[:, data.columns != self.target_col_name], data[self.target_col_name]
            logging.info('Preprocessing data successfully finished')
            return x, y
        except KeyError:
            logging.error('Invalid input data')
            api.abort(400, 'Invalid input data.')

    @metrics.summary('my_custom_summary_fit_model', 'Time of request', labels={'status': lambda resp: resp.status_code})
    def fit(self, model_id, params):
        if self.data is None:
            logging.error('No data found')
            api.abort(400, 'Please provide dataframe before fit the model')
        x, y = self.data_preprocessing(self.data)
        self.train_x_ohe = x
        for mod in self.models:
            if mod['model_id'] == model_id:
                try:
                    logging.info('Fitting model in progress...')
                    if params:
                        model = mod['model'].set_params(**params)
                    else:
                        logging.warning('Default params were set')
                        model = mod['model']
                    with mlflow.start_run(run_name='init_fit'):
                        model.fit(x, y)
                        pickle.dump(model, open(self.model_file_name+"{name}.pickle".format(name=model_id), "wb"))
                        logging.info('Fitting model successfully finished')
                        mlflow.log_params(model.get_params())
                    return {'status_code': 200}
                except ValueError:
                    logging.error('Invalid parameter for estimator {}.'.format(str(mod['model'])))
                    api.abort(400, 'Invalid parameter for estimator {}.'.format(str(mod['model'])))

    def predict(self, model_id, data):
        data = pd.read_json(data)
        if self.data is None:
            logging.error('No data was provided to fit model')
            api.abort(400, 'Please provide dataframe and fit the model before prediction.')
        x, _ = self.data_preprocessing(data)
        ##################################
        _, x = self.train_x_ohe.align(x, join='left', axis=1)
        x.fillna(0, inplace=True)
        ###############################
        for model in self.models:
            if model['model_id'] == model_id:
                try:
                    logging.info('Prediction in progress...')
                    path = os.path.join(self.model_file_name, str(model_id) + '.pickle')
                    model = pickle.load(open(path, 'rb'))
                    prediction = list(model.predict(x))
                    logging.info('Prediction successfully finished')
                    return {'prediction': [int(x) for x in prediction]}
                except FileNotFoundError:
                    logging.error('Fitted model wasnt found')
                    api.abort(400, 'Model {} should be fitted first.'.format(str(model['model'])))
                except ValueError:
                    logging.error('Wrong dim')
                    api.abort(400, 'Dimension mismatch. Check provided data features.')

    def retrain(self, model_id, data):
        try:
            params = data['params']
        except KeyError:
            params = None
        self.post(data)
        logging.info('Retrain in progress...')
        for model in self.models:
            if model['model_id'] == model_id:
                path = os.path.join(self.model_file_name, str(model_id) + '.pickle')
                if os.path.isfile(path):
                    self.fit(model_id, params)
                else:
                    logging.error('Initial model wasnt found')
                    api.abort(400, 'Train initial model with id {} first.'.format(model_id))
        logging.info('Retrain successfully finished')

    def delete(self, model_id):
        try:
            logging.info('Deleting model...')
            path = os.path.join(self.model_file_name, str(model_id) + '.pickle')
            os.remove(path)
            logging.info('Deleting successfully finished')
        except FileNotFoundError:
            logging.error('No model was found')
            api.abort(400, 'No model for model_id {} found.'.format(model_id))


ml_models = MLModelsDAO()


@api.route('/post_data')
class MLModels(Resource):
    """Загрузка данных от пользователя"""
    def get(self):
        return ml_models.get()

    def post(self):
        uploaded_file = request.json
        ml_models.post(uploaded_file)
        return jsonify(reply='Data loaded')


@api.route('/train_model/<int:model_id>')
class MLModels(Resource):
    """Загрузка данных и обучение модели с учетом заданных параметров"""
    def post(self, model_id):
        params = request.json
        ml_models.fit(model_id, params)
        return jsonify(reply='Train finished')



@api.route('/predict/<int:model_id>')
class MLModels(Resource):
    """Предсказание конкретной модели на данных от пользователя"""
    def post(self, model_id):
        data_for_prediction = request.json
        prediction = ml_models.predict(model_id, data_for_prediction)
        return jsonify(prediction)


@api.route('/alter/<int:model_id>')
class MLModels(Resource):
    """Обучение заново и удаление старой модели"""
    def put(self, model_id):
        params = request.json
        ml_models.retrain(model_id, params)
        return jsonify(reply='Retrain finished')

    def delete(self, model_id):
        ml_models.delete(model_id)
        return jsonify(reply='Model was deleted')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8080')