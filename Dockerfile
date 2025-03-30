FROM python:3.12

WORKDIR /app

COPY .venv/Scripts/requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "server.py"]

//Remove this command (For server)
