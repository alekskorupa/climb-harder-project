mlflow:
	mlflow server --host 127.0.0.1 --port 8080

app:
	streamlit run src/app/app.py

model_api:
	python src/api/main.py

database:
	docker run --name postgres -e POSTGRES_PASSWORD= -p 5432:5432 -d postgres
