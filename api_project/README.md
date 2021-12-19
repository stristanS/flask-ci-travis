Порядок действий:

- ### Список доступных для обучения моделей можно посмотреть через: requests.get('http://localhost:8080/post_data')<br />
- ### Загрузка данных происходит через requests.post('http://127.0.0.1:8080/post_data', json = data)<br />
    - При этом передаваемый json имеет следующий формат: {payload: файл.json, target_col_name: str, columns_to_drop: list/None')<br />
        - payload - данные в формате .json для обучения модели. Обязательное поле<br />
        - target_col_name - название колонки таргета, тип string. Обязательное поле<br />
        - columns_to_drop - колонки, которые следует удалить, list или None.<br />
- ### Обучение модели происходит через requests.post('http://localhost:8080/train_model/<id модели>', json = data)<br />
    - <id модели> в пути означает индекс модели из списка 0, т.е. возможные значения 1 (регрессия) или 2 (классификация).<br />
    - При этом передаваемый json - это параметры модели в формате н-р: {'fit_intercept': True. В качестве параметров можно использовать названия параметров для моделей из п.0 из библиотеки sklearn, соблюдая нейминг. Можно не передавать параметры, тогда обучение происходит с дефолтными.<br />
- ### Предсказание модели происходит через requests.post('http://localhost:8080/predict/<id модели>', json = data)<br />
    - <id модели> должен совпадать с указанным при обучении <id модели>.<br />
    - Передаваемый json это данные в формате .json для тестирования модели. Обязательное поле.<br />
- ### Повторное обучение модели происходит через requests.put('http://localhost:8080/alter/<id модели>', json = data)<br />
    - <id модели> в пути означает индекс модели из списка 0, т.е. возможные значения 1 (регрессия) или 2 (классификация). Сначала нужно обучить исходную модель из п.2.<br />
    - Передаваемый json имеет следующий формат:{payload: файл.json, target_col_name: str, columns_to_drop: list/None, params: {}')<br />
        - payload - данные в формате .json для обучения модели. Обязательное поле<br />
        - target_col_name - название колонки таргета, тип string. Обязательное поле<br />
        - columns_to_drop - колонки, которые следует удалить, list или None.<br />
        - params - параметры, с которыми будет переобучаться модель, dict.<br />
- ### Удаление модели происходит через requests.delete('http://localhost:8080/alter/1<id модели>')<br />
    - <id модели> в пути означает индекс модели из списка 0, т.е. возможные значения 1 или 2. Сначала нужно обучить исходную модель из п.2<br />
- ### Логирование
    - логи пишутся в docker volume record.log файл
- ### Мониторинг
    - результаты доступны через ('http://localhost:8080/metrics')
    - Prometheus доступен через ('http://localhost:9090')  
    - для мониторинга используется библиотека prometheus_flask_exporter
    - пишутся стандартные метрики + 2 кастомные для методов get и fit класса MLModelsDAO (написаны просто для проверки, что все ок).
- ### MLflow
    - доступен через ('http://localhost:5000/')
    - используется в методе fit для сохранения параметров модели.
- ### Тесты
    - Тестируется response 200 для двух методов post загрузка данных и и get вывод моделей ('/post_data').

__Для тестирования использовались только датасеты с небольшим количеством наблюдений (до 1000) и признаков.__

Пример праметров для BOSTON DATASET:

    1. payload: 
        1.1. train: pd.read_csv('data/boston_train.csv', index_col = 0).to_json()
        1.2. test: pd.read_csv('data/boston_test.csv', index_col = 0).to_json()
    2. target_col_name: 'medv' 
    3. columns_to_drop: None / любая
    4. <id модели>: 1 
    5. model_params = None / {fit_intercept=True, normalize='deprecated', copy_X=True, n_jobs=None, positive=False}


Пример праметров для TITANIC DATASET:
    
    1. payload: 
        1.1.  train: pd.read_csv('data/titanic_train.csv', index_col = 0).to_json()
        1.2. test: pd.read_csv('data/titanic_test.csv', index_col = 0).to_json()
    3. target_col_name: 'Survived' 
    4. columns_to_drop: None / ['Name', 'Ticket'] (желательно удалить колонки, иначе большое признаковое пространство) 
    5. <id модели>: 2
    6. model_params = None / {'penalty': 'l2', etc}, но строго параметры sklearn.linear_model.LogisticRegression Parameters 