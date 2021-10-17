FROM python:3
ENV PYTHONUNBUFFERED 1

RUN mkdir /app
WORKDIR /app
RUN mkdir ./data
RUN mkdir ./results

COPY /data/fraud_cases.csv /app/data/fraud_cases.csv
COPY /data/FRISS_ClaimHistory_test.csv /app/data/FRISS_ClaimHistory_test.csv
COPY /data/FRISS_ClaimHistory_training.csv /app/data/FRISS_ClaimHistory_training.csv
COPY main.py /app
COPY visualizations.py /app
COPY api.py /app
COPY requirements.txt /app

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt

RUN python3 main.py

EXPOSE 8000
CMD ['uvicorn', 'api:app']
