# Housing Prices

This is a Capstone Project for *[Azure Machine Learning Engineer Nanodegree Program - Udacity](https://www.udacity.com/course/machine-learning-engineer-for-microsoft-azure-nanodegree--nd00333)*.

In this project, we will create two models using a selected dataset: one using Automated ML and one customized model whose hyperparameters are tuned using HyperDrive. We will then compare the performance of both the models and deploy the best performing model.

Housing prices datasets are perfect matches for starting Machine Learning (ML). They can easily be used to learn tabular data, missing values, imputations, encodings, fit and transform mechanisms, feature engineering, etc. As a result, lots of entry-level books and courses about ML use housing prices datasets in their contents.

Consequently, for this capstone project, I have decided to select a housing prices dataset from [Kaggle](https://www.kaggle.com/). There can be found lots of housing prices datasets in Kaggle. But, if some suitable competitions can be found, I can also participate in the competitions with the results of this project. As a result, I have decided to participate in two competitions: *[Housing Prices Competition for Kaggle Learn Users](https://www.kaggle.com/c/home-data-for-ml-course/overview) and [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)*. More details about these competitions and the related datasets will be given in the Dataset section.

My workflow will be as follows:

- Get the datasets,
- Import the datasets to AzureML Studio,
- Create an auto ml experiment and select the best auto ml run,
- Create a hyperdrive experiment and select the best hyperdrive run,
- Compare the runs and select the best run for the project,
- Make a model from the best run,
- Create a submission file from the model,
- Deploy a web service from the model
- Test the model endpoint.

## Project Set-Up and Installation

- Click the green 'Code' button to download the .zip of the files from the GitHub repository,
- Unzip the starter_files folder,
- Access the Azure Machine Learning Workspace. If you don't have an Azure account you can create a free account by clicking the green *Try Azure for free* button in [this link](https://azure.microsoft.com/en-us/). **Beware! you may be charged to complete this project depending on your subscription**.
- Navigate to the *Datasets* page.
- Register the training dataset by clicking on *Create Dataset* button and select *From Local Files* option.
<p style="color:blue;font-size:10px;">Create dataset from local files</p>

![setup-1.png](images/setup/setup-1.png)

- Name the dataset as *'Housing Prices Dataset'*, select *tabular* as the *dataset type* and click next.
<p style="color:blue;font-size:10px;">Naming the Dataset</p>

![dataset-1.png](images/setup/dataset-1.png)

- Browse to the *'./starter_file/data'* folder in your computer and select *train.csv*, click next.

<p style="color:blue;font-size:10px;">Selecting the local file (train.csv)</p>

![dataset-2.png](images/setup/dataset-2.png)

- Select *Use headers from the first file* option in the *Column headers* field.

<p style="color:blue;font-size:10px;">Changing column headers</p>

![dataset-3.png](images/setup/dataset-3.png)

- Find the *GarageYrBlt* feature and change the *Type* field from *String* to *Integer* and click next.

<p style="color:blue;font-size:10px;">Changing Type of 'GarageYrBlt' column</p>

![dataset-4.png](images/setup/dataset-4.png)

- Confirm the details and click the blue *Create* button.

<p style="color:blue;font-size:10px;">Create the training dataset</p>

![dataset-5.png](images/setup/dataset-5.png)

- Follow the same steps and register the test dataset. To do this repeat the previous steps as described below. Beware of the name changes marked as bold!:
    - Click on the *Create Dataset* button and select the *From Local Files* option.
    - Name the dataset as **'Housing Prices Test Dataset'**.
    - Browse to the *'./starter_file/data'* folder in your computer and select **test.csv**.
    - Select *Use headers from the first file* option in the *Column headers*.
    - Find the *GarageYrBlt* feature and change the *Type* field from *String* to *Integer*.
    - Confirm the details and click the blue *Create* button.
- Navigate to Compute page.
- Create a new compute instance by clicking the new button.
- Select virtual machine size as *'Standard_DS3_v2'*. Leave everything as default and click next,
- Give a name to your compute instance and click the *Create* button.
- Navigate to the Notebooks page.
- From the left of the screen create a new folder by clicking the 3 dots next to your user name. Give the name *data* to the folder. 

<p style="color:blue;font-size:10px;">Create data folder</p>

![setup-2.png](images/setup/setup-2.png)

- From the left of the screen select *Upload files* by clicking the 3 dots next to the data folder. Navigate to the *'./starter_file/data'* folder in your local computer and upload *train.csv, test.csv, sample_submission.csv* and *data_description.txt*  files by selecting them so that they can be used if needed. 

<p style="color:blue;font-size:10px;">Upload files to data folder</p>

![setup-3.png](images/setup/setup-3.png)

- From the left of the screen select *Upload files* by clicking the 3 dots next to your user name. Navigate to the *'./starter_file'* folder in your local computer and upload *automl.ipynb, hyperparameter_tuning.ipynb, my-env.yml, score.py* and *train.py* files by selecting them.

<p style="color:blue;font-size:10px;">Upload files</p>

![setup-4.png](images/setup/setup-4.png)

- Click on the *automl.ipynb and hyperparameter_tuning.ipynb* so that they can be opened on the editor.

- Run the notebooks by starting from the first cell one by one. 

## Dataset

### Overview
I will be using the *[Ames Housing dataset](http://jse.amstat.org/v19n3/decock.pdf)* in this project. The original dataset was first published by Dean De Cock in his paper *[Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project](https://www.researchgate.net/publication/267976209_Ames_Iowa_Alternative_to_the_Boston_Housing_Data_as_an_End_of_Semester_Regression_Project) at Journal of Statistics Education (November 2011)*.

The original dataset is used in two different *[Kaggle](https://www.kaggle.com/)* competitions. The first competition is the *[Housing Prices Competition for Kaggle Learn Users](https://www.kaggle.com/c/home-data-for-ml-course/overview)*, and the second competition is the *[House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)* competition.

These are regression competitions in which competitors try to predict the price of the houses in the **test dataset** using the **training dataset**.

The **training dataset** has 1460 rows and 81 columns (including the *Id* field and the target column *SalePrice*). The **test dataset**, on the other hand, has 1459 rows and 80 columns (excluding the target column *SalePrice*).

79 explanatory variables are describing (almost) every aspect of residential homes in Ames, Iowa.

For competition purposes, approximately all of the data has been divided into two parts: **training dataset** and **test dataset**. We will be using the **training dataset** for training and the **test dataset** for submission to the competition. We will also send requests to our deployed web service using the test dataset.

### Task
As written above this is a regression task. We will try to predict the final price of each home in the test set.

I will not explain all the dataset features since they are too many and beyond the scope of this project. However, the explanations can be found in the *[data_description.txt](https://github.com/ErkanHatipoglu/nd00333-capstone/tree/master/starter_file/data/data_description.txt)* file.

I will not use the 'Id' and 'Utilities' features. The former is just the id of the houses and the latter is the same for all items but one.

The target column is *SalePrice*. As mentioned above, this is a regression task project to predict house prices for a given set of features.

There are some missing values in the training and test datasets that are being handled by the clean_data functions. There are two clean_data functions. One is in the *automl.ipynb* and the other is in the *train.py* files. These functions are similar with some minor differences. The clean_data function in the *automl.ipynb* file, for example, transforms the test set, so that we can use it later to predict the house prices and submit to the competitions. I use sklearn transformers and pipelines to preprocess the data in the clean_data functions. 

### Access
The datasets can be downloaded from Kaggle. To download the necessary files just click on one of the competition links above and select the data tab. You may be required to get a free membership and accept the competition rules.

I have already downloaded the datasets to the data folder. Also, I have downloaded the *sample_submission.csv* and *data_description.txt* files. The *sample_submission.csv* file can be used to create submission files, while the *data_description.txt* file may be handy for data analysis.

I have registered the datasets using ML Studio's *Create dataset from local files* feature. The **training dataset** is registered as *Housing Prices Dataset* and the **test dataset** is registered as *Housing Prices Test Dataset*.

<p style="color:blue;font-size:10px;">Registered Datasets</p>

![dataset-8.png](images/dataset/dataset-8.png)

I have loaded these registered datasets as TabularDataset and converted them to pandas dataframes. 

**Warning: *GarageYrBlt* feature should be changed from *String* to *Integer* while creating both datasets. Otherwise, the clean_data function will crash! Kindly refer to *Project Set-Up and Installation* for details**.

## Automated ML
I need to emphasize an important issue first. As I have stated earlier, I have used sklearn transformers and pipelines to preprocess the data in the clean_data functions. Because of those transformations, the column names of the resultant dataframes are being removed. However, my automl configuration needs the target column name. As a result, I have added numeric column names to the resultant dataframes in the notebook. Nothing to be done while reproducing. 

A second issue is the number of columns. Because of the indeterministic nature of my clean_data functions (there is no random_state parameter in the *train_test_split* function), the number of the columns will differ for every run of the notebook if the kernel is restarted. As a result, anyone reproducing the code will probably see different column numbers than this notebook in the repository. But this will not cause any error except for one cell. This cell is only for demonstration purposes and does not affect the automl run. On the other hand, while reproducing, if you want to run all the cells (by clicking the double arrows as shown in the figure below) you need to delete this cell. The cell is shown below:

<p style="color:blue;font-size:10px;">Delete this cell if needed!</p>

![automl-10.png](images/automl/automl-10.png)

As described in [Configure automated ML experiments in Python](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train), there are several options that we can use to configure our automated machine learning experiment. These parameters are set by instantiating an AutoMLConfig object. Below can be found the descriptions and the reasoning for some important parameters:

- *experiment_timeout_minutes*: Maximum amount of time in minutes that all iterations combined can take before the experiment terminates. **We do not want our experiment to cost too much.**

- *max_concurrent_iterations*: Represents the maximum number of iterations that would be executed in parallel. **Since we have *4* nodes in our AmlCompute cluster I have selected 4.**

- *max_cores_per_iteration*: The maximum number of threads to use for a given training iteration. **I have selected *-1* to use all the possible cores per iteration per child-run.** 

- *training_data*: The training data to be used within the experiment (including the target column).

- *validation_data*: The validation data to be used within the experiment (including the target column).

- *label_column_name*: The name of the target column.  

- *enable_early_stopping*: Whether to enable early termination if the score is not improving in the short term. **We do not want our experiment to cost too much.**

- *task*: The type of task to run. For our case, it is **regression**.

- *primary_metric*: The metric that Automated Machine Learning will optimize for model selection. I have chosen *normalized_root_mean_squared_error* as suggested in [this article](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train#primary-metric).

### Results

The best model for the AutoML run is a *VotingEnsemble* by the given metrics below.

- *normalized_root_mean_squared_error:* 0.029610347347248284
- *root_mean_squared_log_error:* 0.11724443714762385
- *mean_absolute_error:* 14028.03885241191

We can find the best model in different ways after the experiment finishes.

- By exploring the *RunDetails* widget using the notebook,

<p style="color:blue;font-size:10px;">RunDetails widget</p>

![automl-8.png](images/automl/automl-8.png)

- By writing code using the notebook,

<p style="color:blue;font-size:10px;">Displaying best model proporties using the notebook</p>

![automl-9.png](images/automl/automl-9.png)

- By exploring the *Experiments* tab using the Azure ML Studio,

<p style="color:blue;font-size:10px;">Completed experiment with the best model summary</p>

![automl-2.png](images/automl/automl-2.png)

The models of the experiment may furtherly be explored by clicking the models tab.

<p style="color:blue;font-size:10px;">AutoML models</p>

![automl-3.png](images/automl/automl-3.png)

We can also click on the best performing model and explore the details of that model.

<p style="color:blue;font-size:10px;">Best performing AutoML model</p>

![automl-4.png](images/automl/automl-4.png)

We can explore the model explanation and the metrics we are interested in. Other than the primary metric, we are interested in the mean absolute error and the root mean squared log error since these are the metrics used in the competitions given above.

<p style="color:blue;font-size:10px;">Model explanation</p>

![automl-11.png](images/automl/automl-11.png)

<p style="color:blue;font-size:10px;">Metrics</p>

![automl-5.png](images/automl/automl-5.png)

The metrics of the best-performing model can also be explored in the related run's view.

<p style="color:blue;font-size:10px;">Best performing AutoML model with Metrics</p>

![automl-6.png](images/automl/automl-6.png)

We can run new AutoML experiments to improve our model. Because of the indeterministic nature of our model with every new run, we get different results.

We can also make some feature engineering before the preprocessing step or after exploring the feature importance tab.

<p style="color:blue;font-size:10px;">Feature importance for the best model</p>

![automl-12.png](images/automl/automl-12.png)

Since we already know the estimator for the best performing model (kindly refer to automl.ipynb for more details), we can try to tune the model, add new models to the voting regressor or retrain the model without validation set (this may reduce the chance of overfitting).

<p style="color:blue;font-size:10px;">Estimator for the best performing model</p>

![automl-9.png](images/automl/automl-9.png)

Since the automl model is better than the hyperdrive model, I have used it to create a submission file for the competitions. This file can be found in the data folder as *submission.csv*. My submissions and the results are as follows:

<p style="color:blue;font-size:10px;">Housing Prices Competition for Kaggle Learn Users - Submission</p>

![kaggle-1.png](images/kaggle/kaggle-1.png)

<p style="color:blue;font-size:10px;">Housing Prices Competition for Kaggle Learn Users - Result</p>

![kaggle-2.png](images/kaggle/kaggle-2.png)

<p style="color:blue;font-size:10px;">House Prices - Advanced Regression Techniques - Submission</p>

![kaggle-3.png](images/kaggle/kaggle-3.png)

<p style="color:blue;font-size:10px;">House Prices - Advanced Regression Techniques - Result</p>

![kaggle-4.png](images/kaggle/kaggle-4.png)

The results are as follows:

- MAE (Housing Prices Competition for Kaggle Learn Users): 14058.80470
- RMSLE (House Prices - Advanced Regression Techniques): 0.12774

The mean absolute error result is similar to the validation result (MAE: 14028.03885) but the root mean squared log result is much higher than the validation result (RMSLE: 0.11724). It seems that I am overfitting the data. I need to improve my model to reduce overfitting. Adding more data to the training set and/or making some feature engineering will probably improve the model.

## Hyperparameter Tuning
## Hyperdrive Configuration

I will use [GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) from [scikit-learn](https://scikit-learn.org/stable/index.html) library. This is a Gradient Boosting for regression. Despite the *[No free lunch theorem](https://en.wikipedia.org/wiki/No_free_lunch_theorem)*, from my previous experience, I believe that gradient boosting gives good results for this dataset. Detailed explanation for *Gradient Boosting* can be found in this [User Guide](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting). Some important parameters for GradientBoostingRegressor are:

- learning_rate(float), default=0.1

   Learning rate shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators.

- n_estimators (int), default=100

   The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.

In addition to their importance, these parameters can be used as an example for discrete hyperparameters (choice) and continuous hyperparameters (Uniform).

As can be seen in [Hyperparameter tuning a model with Azure Machine Learning](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters) we need to

- Define the parameter search space,
- Specify a primary metric to optimize,
- Specify early termination policy for low-performing runs,

Since the learning rate is type float, a continuous search space needs to be selected. uniform() returns a value uniformly distributed between low and high which is used for learning rate. The range for the search space (for the learning rate) is between *0.001* and *0.1*. *0.1* is the default value for the learning rate. As explained above there is a trade-off between the learning rate and the number of estimators. This means that we need to decrease the learning rate to increase the number of estimators. Since I want to increase the number of estimators (explained below), I made the default value to be the maximum of the search space (high) and tried to decrease the learning rate as much as possible by setting the low value to *0.001*.

Since the number of estimators is type int, a discrete search space needs to be selected. Discrete hyperparameters are specified as a choice() among discrete values. The range for the search space (for the number of estimators) is between *100* and *500*. *100* is the default value for the number of estimators. As explained above increasing the number of estimators may decrease overfitting. As a result, made the default value be the minimum of the search space (low) and tried to increase the number of estimators as much as possible by setting the high value to *500*.

Random sampling supports discrete and continuous hyperparameters. It supports the early termination of low-performance runs. As a result, it will be a good choice for parameter sampling.

For our AutoML experiment, the primary metric is chosen to be **normalized_root_mean_squared_error** as suggested in [this article](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train#primary-metric). Since we have to compare the two models for deployment the same primary metric must be chosen for hyperdrive run. Low values are better for this metric.

We can automatically terminate poorly performing runs with an early termination policy. Early termination improves computational efficiency. Bandit policy is based on slack factor/slack amount and evaluation interval. Bandit terminates runs where the primary metric is not within the specified slack factor/slack amount compared to the best performing run. We can select Bandit policy as an early termination policy for more aggressive savings.

Since the estimator class is deprecated and gives an error while calculating mean_squared_error, a ScriptRunConfig object has been used as suggested [here](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.estimator.estimator?view=azure-ml-py). Also an environment is created to remove the mean_squared_error calculation error.

max_concurrent_runs has been selected as *4* since the AmlCompute cluster has 4 nodes.

max_total_runs has been selected as *40* for more runs.

A training script is needed for training. *train.py* is used as a training script and can be found in the starter_files folder of the repository. Sklearn pipelines and transformers are used for preprocessing in the training script since there are missing values in the dataset.

### Results

The best model for the Hyperdrive run has the arguments below:

- learning rate: 0.0925131569584366
- number of estimators: 100

The best model for the Hyperdrive run has the metrics below:

- *normalized_root_mean_squared_error:* 0.032666490716132174
- *root_mean_squared_log_error:* 0.1279803797267253
- *mean_absolute_error:* 15671.028999405176

These metrics are worse than the AutoML model. As a result, I have used the AutoML model for deployment.

We can watch the status of the Hyperdrive run within the notebook.

<p style="color:blue;font-size:10px;">Execution Summary</p>

![hyperdrive-1.png](images/hyperdrive/hyperdrive-1.png)

<p style="color:blue;font-size:10px;">RunDetails Widget</p>

![hyperdrive-2.png](images/hyperdrive/hyperdrive-2.png)

<p style="color:blue;font-size:10px;">Best Model</p>

![hyperdrive-10.png](images/hyperdrive/hyperdrive-10.png)

We can also watch the status of the Hyperdrive run by using the AzureML Studio.

<p style="color:blue;font-size:10px;">Run details</p>

![hyperdrive-3.png](images/hyperdrive/hyperdrive-3.png)

<p style="color:blue;font-size:10px;">Child Runs</p>

![hyperdrive-4.png](images/hyperdrive/hyperdrive-4.png)

<p style="color:blue;font-size:10px;">Best Model Details</p>

![hyperdrive-5.png](images/hyperdrive/hyperdrive-5.png)

<p style="color:blue;font-size:10px;">Best Model Metrics</p>

![hyperdrive-6.png](images/hyperdrive/hyperdrive-6.png)

<p style="color:blue;font-size:10px;">Hyperdrive Experiment</p>

![hyperdrive-7.png](images/hyperdrive/hyperdrive-7.png)

<p style="color:blue;font-size:10px;">Registered Hyperdrive Model</p>

![hyperdrive-9.png](images/hyperdrive/hyperdrive-9.png)

We can try new hyperdrive runs to improve the model. We already know that our model is overfitting the data (from the Kaggle competitions results). It seems that because of the random sampling we have a small number of estimators value for the best model. Since we are overfitting the data, a higher number of estimators value may decrease overfitting. As a result, rerunning the experiment and increasing the number of estimators search space will probably improve the model.

## Model Deployment

- *normalized_root_mean_squared_error*
    - *AutoML Model:* 0.029610347347248284
    - *Hyperdrive Model:* 0.032666490716132174
- *root_mean_squared_log_error*
    - *AutoML Model* 0.11724443714762385
    - *Hyperdrive Model:* 0.1279803797267253
- *mean_absolute_error*
    - *AutoML Model:* 14028.03885241191
    - *Hyperdrive Model:* 15671.028999405176

As can be seen, the AutoML model gives better results. As a result, I have deployed the AutoML model as a web service.

I have registered the best AutoML run for feature uses.

<p style="color:blue;font-size:10px;">Registered AutoML Model</p>

![automl-13.png](images/automl/automl-13.png)

We need an environment to deploy the models. We can both define a custom environment or use an Azure curated environment.

I have first created an Azure curated AutoML environment. Then by using the dependencies of this environment I have generated a *.yml* file. By using this *.yml* file I have created a second environment that I used for deployment. By doing this I have both showed to create a custom environment or use an Azure curated environment. Besides, I have fulfilled the project requirements.

We also need a scoring script (entry) for deployment. This script (score.py) can be found in the starter_files folder of the repository. In this script an init() function that loads the model and a run() function that predicts the requested query must be defined. The input to the run() function is a JSON string.

Finally, we need to define a deployment configuration.

<p style="color:blue;font-size:10px;">Model Deployment Code</p>

![deploy-2.png](images/deploy/deploy-2.png)

<p style="color:blue;font-size:10px;">Model Endpoint - Transitioning State</p>

![deploy-1.png](images/deploy/deploy-1.png)

<p style="color:blue;font-size:10px;">Model Endpoint - Healthy State</p>

![deploy-3.png](images/deploy/deploy-3.png)

<p style="color:blue;font-size:10px;">Endpoints tab</p>

![deploy-4.png](images/deploy/deploy-4.png)

<p style="color:blue;font-size:10px;">Application insights link</p>

![deploy-5.png](images/deploy/deploy-5.png)

To get a [JSON 200](http://enocean-gateway.eu/images/documents/Documentation/error-codes.html) answer (returned when an application request is successful) from the web service we need to create a JSON string. This string needs to be a dictionary with a key *'data'*. The value of this key is a list of dictionaries. Each inner dictionary has keys of column names and values corresponding to the related item that is being queried. An example JSON String for one item (house) is as follows.

```
{"data": [{"0": -0.7877776601303935, "1": 0.40472075580786326, "2": -0.824862038820425, "3": -0.12479112703105394, "4": -0.22231727917043134, "5": 0.5900020843135478, "6": -1.0518023616782892, "7": -0.07341187239855226, "8": -0.7010384200582758, "9": -0.9372116952221238, "10": 1.177698492303729, "11": -0.12402884834943756, "12": -0.7249813707482863, "13": -0.8685999131096809, "14": -0.2597453608736428, "15": 1.8877413397414555, "16": -0.11750960306848886, "17": 0.034213835019685064, "18": 0.07970023247881027, "19": -1.177925864816224, "20": -1.1456169750686023, "21": -0.36519377147345083, "22": -0.34676496051017097, "23": -0.4089140915554155, "24": -0.7687342984741533, "25": 0.34566371578061267, "26": -1.0380498416918418, "27": -0.7881555347992615, "28": -0.07907797829369968, "29": -1.0708264505936433, "30": -0.9354464621967534, "31": 1.6770151821776618, "32": -0.6650718421533498, "33": 0.186221289050153, "34": -0.5834629514310187, "35": 0.0, "36": 0.0, "37": 0.0, "38": 0.0, "39": 1.0, "40": 0.0, "41": 0.0, "42": 0.0, "43": 0.0, "44": 1.0, "45": 0.0, "46": 0.0, "47": 1.0, "48": 0.0, "49": 0.0, "50": 0.0, "51": 0.0, "52": 1.0, "53": 0.0, "54": 0.0, "55": 0.0, "56": 0.0, "57": 0.0, "58": 0.0, "59": 0.0, "60": 1.0, "61": 0.0, "62": 0.0, "63": 0.0, "64": 1.0, "65": 0.0, "66": 0.0, "67": 0.0, "68": 0.0, "69": 0.0, "70": 0.0, "71": 0.0, "72": 0.0, "73": 0.0, "74": 0.0, "75": 0.0, "76": 1.0, "77": 0.0, "78": 0.0, "79": 0.0, "80": 1.0, "81": 0.0, "82": 0.0, "83": 0.0, "84": 0.0, "85": 0.0, "86": 1.0, "87": 0.0, "88": 0.0, "89": 0.0, "90": 0.0, "91": 0.0, "92": 0.0, "93": 0.0, "94": 0.0, "95": 1.0, "96": 0.0, "97": 0.0, "98": 0.0, "99": 0.0, "100": 0.0, "101": 0.0, "102": 1.0, "103": 0.0, "104": 0.0, "105": 0.0, "106": 0.0, "107": 0.0, "108": 0.0, "109": 0.0, "110": 0.0, "111": 1.0, "112": 0.0, "113": 0.0, "114": 0.0, "115": 0.0, "116": 0.0, "117": 0.0, "118": 0.0, "119": 0.0, "120": 0.0, "121": 0.0, "122": 0.0, "123": 0.0, "124": 0.0, "125": 0.0, "126": 0.0, "127": 0.0, "128": 0.0, "129": 0.0, "130": 0.0, "131": 0.0, "132": 0.0, "133": 0.0, "134": 0.0, "135": 0.0, "136": 0.0, "137": 0.0, "138": 0.0, "139": 0.0, "140": 0.0, "141": 0.0, "142": 0.0, "143": 0.0, "144": 0.0, "145": 0.0, "146": 0.0, "147": 0.0, "148": 0.0, "149": 0.0, "150": 0.0, "151": 0.0, "152": 0.0, "153": 0.0, "154": 0.0, "155": 0.0, "156": 0.0, "157": 0.0, "158": 0.0, "159": 0.0, "160": 0.0, "161": 0.0, "162": 0.0, "163": 0.0, "164": 0.0, "165": 0.0, "166": 0.0, "167": 0.0, "168": 0.0, "169": 0.0, "170": 0.0, "171": 0.0, "172": 0.0, "173": 0.0, "174": 0.0, "175": 0.0, "176": 0.0, "177": 0.0, "178": 0.0, "179": 0.0, "180": 0.0, "181": 0.0, "182": 0.0, "183": 0.0, "184": 0.0, "185": 0.0, "186": 0.0, "187": 0.0, "188": 0.0, "189": 0.0, "190": 0.0, "191": 0.0, "192": 0.0, "193": 0.0, "194": 0.0, "195": 0.0, "196": 1.0, "197": 0.0, "198": 0.0, "199": 0.0, "200": 0.0, "201": 0.0, "202": 0.0, "203": 0.0, "204": 0.0, "205": 0.0, "206": 0.0, "207": 0.0, "208": 0.0, "209": 0.0, "210": 0.0, "211": 0.0, "212": 0.0, "213": 0.0, "214": 0.0, "215": 0.0, "216": 0.0, "217": 0.0, "218": 0.0, "219": 0.0, "220": 0.0, "221": 1.0, "222": 0.0, "223": 1.0, "224": 0.0, "225": 0.0, "226": 0.0, "227": 0.0, "228": 0.0, "229": 0.0, "230": 0.0, "231": 0.0, "232": 0.0, "233": 0.0, "234": 0.0, "235": 0.0, "236": 1.0, "237": 1.0, "238": 0.0, "239": 0.0, "240": 0.0, "241": 0.0, "242": 0.0, "243": 0.0, "244": 0.0, "245": 0.0, "246": 1.0, "247": 0.0, "248": 0.0, "249": 0.0, "250": 0.0, "251": 0.0, "252": 0.0, "253": 0.0, "254": 0.0, "255": 0.0, "256": 0.0, "257": 0.0, "258": 0.0, "259": 1.0, "260": 0.0, "261": 0.0, "262": 0.0, "263": 0.0, "264": 1.0, "265": 0.0, "266": 0.0, "267": 1.0, "268": 0.0, "269": 0.0, "270": 0.0, "271": 0.0, "272": 0.0, "273": 1.0, "274": 0.0, "275": 1.0, "276": 0.0, "277": 0.0, "278": 0.0, "279": 1.0, "280": 0.0, "281": 0.0, "282": 0.0, "283": 0.0, "284": 0.0, "285": 0.0, "286": 0.0, "287": 0.0, "288": 0.0, "289": 0.0, "290": 0.0, "291": 0.0, "292": 1.0, "293": 0.0, "294": 0.0, "295": 0.0, "296": 0.0, "297": 0.0, "298": 0.0, "299": 0.0, "300": 0.0, "301": 0.0, "302": 0.0, "303": 0.0, "304": 0.0, "305": 0.0, "306": 0.0, "307": 0.0, "308": 0.0, "309": 1.0, "310": 0.0, "311": 1.0, "312": 0.0, "313": 0.0, "314": 0.0, "315": 0.0, "316": 0.0, "317": 0.0, "318": 0.0, "319": 0.0, "320": 1.0, "321": 0.0, "322": 0.0, "323": 0.0, "324": 0.0, "325": 1.0, "326": 0.0, "327": 0.0, "328": 0.0, "329": 0.0, "330": 0.0, "331": 1.0, "332": 0.0, "333": 0.0, "334": 0.0, "335": 0.0, "336": 1.0, "337": 0.0, "338": 0.0, "339": 0.0, "340": 0.0, "341": 1.0, "342": 0.0, "343": 0.0, "344": 0.0, "345": 0.0, "346": 0.0, "347": 1.0, "348": 0.0, "349": 0.0, "350": 0.0, "351": 0.0, "352": 1.0, "353": 0.0, "354": 0.0, "355": 0.0, "356": 0.0, "357": 0.0, "358": 0.0, "359": 1.0, "360": 0.0, "361": 0.0, "362": 0.0, "363": 1.0, "364": 0.0, "365": 0.0, "366": 0.0, "367": 0.0, "368": 0.0, "369": 0.0, "370": 0.0, "371": 0.0, "372": 1.0, "373": 0.0, "374": 0.0, "375": 0.0, "376": 0.0, "377": 0.0, "378": 1.0, "379": 0.0, "380": 0.0, "381": 0.0, "382": 0.0, "383": 0.0, "384": 1.0, "385": 0.0, "386": 0.0, "387": 0.0, "388": 1.0, "389": 0.0, "390": 0.0, "391": 1.0, "392": 0.0, "393": 0.0, "394": 0.0, "395": 1.0, "396": 0.0, "397": 0.0, "398": 0.0}]}

```
As described above, this JSON string is only for the model (automl_model.pkl) which is in the *'./starter_file'* folder of the repository. For every new model (if the project is reproduced) the JSON string will be slightly different because of the varying column numbers. The related cells of the notebook **will handle this automatically**. 

The code that handles JSON string for one item is as follows:

```
# Request for 1 item
# Get the test data
my_test_values = test_dataset.to_pandas_dataframe()
test_list = my_test_values.values.tolist()

# Get first item
my_list=test_list[0]

# Create JSON string
my_data={}
print("Length: ")
print(len(my_list))

# Create dictionary
for count in range(len(my_list)):
    my_data[str(count)]=my_list[count]
print("My Data: ")
print(my_data)

# Convert to JSON format

data = {"data":
        [
          my_data,
      ]
    }

print("Data: ")
print(data)

# Convert to JSON string
input_data = json.dumps(data)
print("Input data")
print(input_data)
```

The code that handles JSON string for 3 items is as follows:

```
# Request for 3 items
my_list_2 = my_test_values.values.tolist()
my_list=[my_list_2[0], my_list_2[1], my_list_2[2]]
test_list=[]
# Create JSON string
for item in my_list:
    # Create dictionary
    my_data={}
    for count in range(len(item)):
        my_data[str(count)]=item[count]
    test_list.append(my_data)
data = {"data":
        test_list
    }
# Convert to JSON string
input_data = json.dumps(data)
```

*test_dataset* is a tabular dataset that is created from *test.csv* within the notebook.

The response of the web service for the first item in the *test.csv* file is as follows:

<p style="color:blue;font-size:10px;">Response of the web service for one item</p>

![deploy-6.png](images/deploy/deploy-6.png)

The response of the web service for the first 3 item in the *test.csv* file is as follows:

<p style="color:blue;font-size:10px;">Response of the web service for three items</p>

![deploy-7.png](images/deploy/deploy-7.png)

These values can also be compared to the first 3 items in the *submission.csv* file that is predicted by the model above.

## Screen Recording
[Project Screencast](https://www.youtube.com/watch?v=vO6-I2WSCMA)

Thank you for reading.
