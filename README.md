[![Build Status](https://app.travis-ci.com/stristanS/flask-ci-travis.svg?branch=master)](https://app.travis-ci.com/stristanS/flask-ci-travis)

Структура проекта.

    .
    ├── api_project
    │   ├── data          		        # .csv файлы для обучения/предсказания
    │   ├── api.py    			# flask - приложение для загрузки данных, обучения и предсказания
    │   ├── Dockerfile    	
    │   ├── example_http_requests.py  	# запуск всех методов flask - приложения
    │   ├── README.md    
    │   ├── requirements.txt    
    │   └── test.py                         # тестирование unittest
    ├── mlflow
    │   └── Dockerfile                      # mlflow для управления экспериментами
    ├── storage				
    ├── .travis.yml
    ├── docker-compose.yml
    ├── prometheus.yml
    ├── requirements-dev.txt
    └── README.md

Для запуска проекта: 
- Клонировать проект, перейти в папку проекта и запустить в терминале следующие команды:
	- docker-compose build
	- docker-compose up
    
- После выполнения команд запускаются три контейнера:
    - flask приложение доступно через 'http://localhost:8080/'
    - mlflow доступно через 'http://localhost:5000/'
    - prometheus доступно через 'http://localhost:9090/', а также можно посмотреть сами метрики через 'http://localhost:8080/metrics'
    
- Для запуска всех методов приложения (более детальное описание в api_project/README) необходимо запустить файл example_http_requests.py, который последовательно запускает все методы:
	- загрузку данных 
	- вывод доступных для обучения моделей
	- обучение 
	- предсказание 
	- переобучение модели
	- удаление обученной модели
    
- Логирование:
    - реализовано с помощью logging
    - логи пишутся в docker volume record.log файл
- Мониторинг:
    - результаты доступны через ('http://localhost:8080/metrics')
    - Prometheus доступен через ('http://localhost:9090')  
    - для мониторинга используется библиотека prometheus_flask_exporter
    - пишутся стандартные метрики + 2 кастомные для методов get и fit класса MLModelsDAO (написаны просто для проверки, что все ок).
- MLflow
    - доступен через ('http://localhost:5000/')
    - используется в методе fit для сохранения параметров модели.
- Тесты
    - Тестируется response 200 для двух методов post (загрузка данных) и и get (вывод моделей) ('/post_data').
- CI 
    - Запускает тесты
    - Собирает docker образы и публикует на docker hub
