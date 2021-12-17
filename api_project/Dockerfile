FROM python:3.8

COPY . /api_project
WORKDIR /api_project

EXPOSE 8080

RUN pip3 install -r requirements.txt

CMD ["python", "-u", "api.py"]