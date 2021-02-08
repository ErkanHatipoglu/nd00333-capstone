import joblib
import numpy as np
import pandas as pd
import os
import json

# The init() method is called once, when the web service starts up.
#
# Typically you would deserialize the model file, as shown here using joblib,
# and store it in a global variable so your run() method can access it later.
def init():
    global model

    # The AZUREML_MODEL_DIR environment variable indicates
    # a directory containing the model file you registered.
    model_filename = 'automl_model.pkl'
    model_path = os.path.join(os.environ['AZUREML_MODEL_DIR'], model_filename)

    model = joblib.load(model_path)


# The run() method is called each time a request is made to the scoring API.
# https://knowledge.udacity.com/questions/442907
def run(data):
    print("data before")
    print(data)
    print(type(data))
    try:
        data = json.loads(data)['data']
        data = pd.DataFrame.from_dict(data)
        print("Dataframe: ")
        print(data.head())
        # Use the model object loaded by init().
        result = model.predict(data)
        print("Result: ")
        print(result)
        # You can return any JSON-serializable object.
        return result.tolist()

    except Exception as e:
        result = str(e)
        # return error message back to the client
        return json.dumps({"error": result})
