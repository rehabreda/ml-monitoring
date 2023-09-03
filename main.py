


from fastapi import FastAPI
import mlflow
from pydantic import BaseModel
import psycopg
import joblib
import datetime

from prefect import task, flow
import pandas as pd
import psycopg2 
from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric
import os 


mlflow.set_tracking_uri("http://localhost:5000")
app = FastAPI()





def load_model(model_uri):
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    return loaded_model


def get_data_from_db(table_name):
    connection = psycopg2.connect(host='localhost', user='postgres',password='example',database='test', port=5432)

    df = pd.read_sql(f"select * from  {table_name} ", con=connection)
    return df 





def calculate_metrics_postgresql(curr,reference_data,current_data):

    

    num_features = list(reference_data.columns[:-1])

    column_mapping = ColumnMapping(
        prediction='prediction',
        numerical_features=num_features,
        categorical_features=None,
        target=None
    )

    report = Report(metrics = [
        ColumnDriftMetric(column_name='prediction'),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric()
    ])

    report.run(reference_data = reference_data, current_data = current_data, column_mapping=column_mapping)
    result = report.as_dict()
    prediction_drift = result['metrics'][0]['result']['drift_score']
    num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
    share_missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']
    curr.execute(
		"insert into predictions_metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_values) values (%s, %s, %s, %s)",
		( datetime.datetime.now(), prediction_drift, num_drifted_columns, share_missing_values)
	)







class InputData(BaseModel):
    # Define your input data fields here, for example
    pixel_0_0: float
    pixel_0_1: float
    pixel_0_2: float
    pixel_0_3: float
    pixel_0_4: float
    pixel_0_5: float
    pixel_0_6: float
    pixel_0_7: float
    pixel_1_0: float
    pixel_1_1: float
    pixel_1_2: float
    pixel_1_3: float
    pixel_1_4: float
    pixel_1_5: float
    pixel_1_6: float
    pixel_1_7: float
    pixel_2_0: float
    pixel_2_1: float
    pixel_2_2: float
    pixel_2_3: float
    pixel_2_4: float
    pixel_2_5: float
    pixel_2_6: float
    pixel_2_7: float
    pixel_3_0: float
    pixel_3_1: float
    pixel_3_2: float
    pixel_3_3: float
    pixel_3_4: float
    pixel_3_5: float
    pixel_3_6: float
    pixel_3_7: float
    pixel_4_0: float
    pixel_4_1: float
    pixel_4_2: float
    pixel_4_3: float
    pixel_4_4: float
    pixel_4_5: float
    pixel_4_6: float
    pixel_4_7: float
    pixel_5_0: float
    pixel_5_1: float
    pixel_5_2: float
    pixel_5_3: float
    pixel_5_4: float
    pixel_5_5: float
    pixel_5_6: float
    pixel_5_7: float
    pixel_6_0: float
    pixel_6_1: float
    pixel_6_2: float
    pixel_6_3: float
    pixel_6_4: float
    pixel_6_5: float
    pixel_6_6: float
    pixel_6_7: float
    pixel_7_0: float
    pixel_7_1: float
    pixel_7_2: float
    pixel_7_3: float
    pixel_7_4: float
    pixel_7_5: float
    pixel_7_6: float
    pixel_7_7: float

@app.post("/predict/")
def predict(data: InputData):
    # Extract input data from 'data', process it if necessary
    input_data = [[data.pixel_0_0,
                    data.pixel_0_1,
                    data.pixel_0_2,
                    data.pixel_0_3,
                    data.pixel_0_4,
                    data.pixel_0_5,
                    data.pixel_0_6,
                    data.pixel_0_7,
                    data.pixel_1_0,
                    data.pixel_1_1,
                    data.pixel_1_2,
                    data.pixel_1_3,
                    data.pixel_1_4,
                    data.pixel_1_5,
                    data.pixel_1_6,
                    data.pixel_1_7,
                    data.pixel_2_0,
                    data.pixel_2_1,
                    data.pixel_2_2,
                    data.pixel_2_3,
                    data.pixel_2_4,
                    data.pixel_2_5,
                    data.pixel_2_6,
                    data.pixel_2_7,
                    data.pixel_3_0,
                    data.pixel_3_1,
                    data.pixel_3_2,
                    data.pixel_3_3,
                    data.pixel_3_4,
                    data.pixel_3_5,
                    data.pixel_3_6,
                    data.pixel_3_7,
                    data.pixel_4_0,
                    data.pixel_4_1,
                    data.pixel_4_2,
                    data.pixel_4_3,
                    data.pixel_4_4,
                    data.pixel_4_5,
                    data.pixel_4_6,
                    data.pixel_4_7,
                    data.pixel_5_0,
                    data.pixel_5_1,
                    data.pixel_5_2,
                    data.pixel_5_3,
                    data.pixel_5_4,
                    data.pixel_5_5,
                    data.pixel_5_6,
                    data.pixel_5_7,
                    data.pixel_6_0,
                    data.pixel_6_1,
                    data.pixel_6_2,
                    data.pixel_6_3,
                    data.pixel_6_4,
                    data.pixel_6_5,
                    data.pixel_6_6,
                    data.pixel_6_7,
                    data.pixel_7_0,
                    data.pixel_7_1,
                    data.pixel_7_2,
                    data.pixel_7_3,
                    data.pixel_7_4,
                    data.pixel_7_5,
                    data.pixel_7_6,
                    data.pixel_7_7]]

    # Make predictions using the loaded MLflow model
    #model_uri = 'runs:/e732c2436d6f4893932c0d0555fb1036/mlruns'
    model_uri = os.getenv('model_uri')
    loaded_model=load_model(model_uri)
    reference_data = get_data_from_db('reference')
    current_data=pd.DataFrame(input_data)
    current_data.columns=list(reference_data.columns[:-1])
    current_data['prediction']=loaded_model.predict(input_data)[0]
   
    with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example", autocommit=True) as conn:
            with conn.cursor() as curr:
                calculate_metrics_postgresql(curr,reference_data,current_data)

       


    return {"predictions": current_data['prediction'].tolist()[0]}
