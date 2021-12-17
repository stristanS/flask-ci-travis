Порядок действий:

0. Список доступных для обучения моделей можно посмотреть через: requests.get('http://127.0.0.1:5000/post_data')

1. Загрузка данных происходит через requests.post('http://127.0.0.1:5000/post_data', json = data)
    При этом передаваемый json имеет следующий формат: {payload: файл.json, target_col_name: str, columns_to_drop: list/None')
        payload - данные в формате .json для обучения модели. Обязательное поле
        target_col_name - название колонки таргета, тип string. Обязательное поле
        columns_to_drop - колонки, которые следует удалить, list или None.
        
2. Обучение модели происходит через requests.post('http://127.0.0.1:5000/train_model/<id модели>', json = data)
    <id модели> в пути означает индекс модели из списка 0, т.е. возможные значения 1 (регрессия) или 2 (классификация). 
    При этом передаваемый json - это параметры модели в формате н-р: {'fit_intercept': True}
    В качестве параметров можно использовать названия параметров для моделей из п.0 из библиотеки sklearn, соблюдая нейминг.
    Можно не передавать параметры, тогда обучение происходит с дефолтными.

3. Предсказание модели происходит через requests.post('http://127.0.0.1:5000/predict/<id модели>', json = data)
    <id модели> должен совпадать с указанным при обучении <id модели>. 
    Передаваемый json это данные в формате .json для тестирования модели. Обязательное поле.
    
4. Повторное обучение модели происходит через requests.put('http://127.0.0.1:5000/alter/<id модели>', json = data)
    <id модели> в пути означает индекс модели из списка 0, т.е. возможные значения 1 (регрессия) или 2 (классификация). 
    Сначала нужно обучить исходную модель из п.2
    Передаваемый json имеет следующий формат:{payload: файл.json, target_col_name: str, columns_to_drop: list/None, params: {}')
        payload - данные в формате .json для обучения модели. Обязательное поле
        target_col_name - название колонки таргета, тип string. Обязательное поле
        columns_to_drop - колонки, которые следует удалить, list или None.
        params - параметры, с которыми будет переобучаться модель, dict.

5. Удаление модели происходит через requests.delete('http://127.0.0.1:5000/alter/1<id модели>')
    <id модели> в пути означает индекс модели из списка 0, т.е. возможные значения 1 или 2. 
    Сначала нужно обучить исходную модель из п.2 


Для тестирования использовались только toy датасеты с небольшим количеством наблюдений (до 1000) и признаков.

Пример праметров для BOSTON DATASET:
    payload: 
        - train: pd.read_csv('data/boston_train.csv', index_col = 0).to_json()
        - test: pd.read_csv('data/boston_test.csv', index_col = 0).to_json()
    target_col_name: 'medv' 
    columns_to_drop: None / любая
    <id модели>: 1 
    model_params = None / {fit_intercept=True, normalize='deprecated', copy_X=True, n_jobs=None, positive=False}


Пример праметров для TITANIC DATASET:
    payload: 
        - train: pd.read_csv('data/titanic_train.csv', index_col = 0).to_json()
        - test: pd.read_csv('data/titanic_test.csv', index_col = 0).to_json()
    target_col_name: 'Survived' 
    columns_to_drop: None / ['Name', 'Ticket'] (желательно удалить колонки, иначе большое признаковое пространство) 
    <id модели>: 2
    model_params = None / {'penalty': 'l2', etc}, но строго параметры sklearn.linear_model.LogisticRegression Parameters 

 
Пример праметров для BREST CANCER DATASET:
    payload: 
        train: pd.read_csv('data/brest_canser_train.csv', index_col = 0).to_json()
        test: pd.read_csv('data/brest_canser_test.csv', index_col = 0).to_json()
    target_col_name: 'diagnosis' 
    columns_to_drop: None / ['id'] (желательно удалить) 
    <id модели>: 2
    model_params = None / {'penalty': 'l2', etc}, но строго параметры sklearn.linear_model.LogisticRegression Parameters 
    