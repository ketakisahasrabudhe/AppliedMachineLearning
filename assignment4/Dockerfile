FROM python:3.9-slim

WORKDIR /app

# install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the app code and saved model
COPY score.py .
COPY app.py .
COPY best_model.pkl .

# expose the port flask runs on
EXPOSE 5000

# run the flask app on container start
CMD ["python", "app.py"]
