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
#
def run(data):
    print("data before")
    print(data)
    data=NumpyParameterType(np.array([data])))
    print("data after")
    print(data)
    names = [i for i in range(398)]
    df = pd.DataFrame(data=np.concatenate(data), index=None, columns=names)
    print("df:")
    print(df.head)
    try:
        #data=np.array(data)
        names = [i for i in range(398)]
        df = pd.DataFrame(data=data, index=None, columns=names)
        print("df:")
        print(df.head)


        # Use the model object loaded by init().
        result = model.predict(df)
        print(result)
        # You can return any JSON-serializable object.
        return result.tolist()
    except Exception as e:
        result = str(e)
        # return error message back to the client
        return json.dumps({"error": result})
