
import requests
import psycopg2 
import pandas as pd 
from time import sleep 

def get_data_from_db(table_name):
    connection = psycopg2.connect(host='localhost', user='postgres',password='example',database='test', port=5432)

    df = pd.read_sql(f"select * from  {table_name} ", con=connection)
    return df 

def simulate_poduction():
    api_url = "http://localhost:8000/predict/"
    prod_data=get_data_from_db('production')
    for i in range(len(prod_data))  :
        response = requests.post(api_url, json=prod_data.iloc[i].to_dict())
        data = response.json()
        predictions = data.get("predictions")
        print("Predictions:", predictions)
        sleep(120)


if __name__ == "__main__":
    simulate_poduction()

