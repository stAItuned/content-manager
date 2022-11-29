---
title: Model deployment
author: Gianluca La Malfa
date: 
topics: []
meta: How many times did you build up a great machine learning model that never seen the light? This is the right article for you!
target: Expert
language: English
cover: FastAPI.png
---

# Model Deployment

How many times did you build up a great machine learning model that never seen the light? This is the right article for you!

Model deployment is the process of putting a machine learning or deep learning algorithm into production. This makes it accessible to users which can work with it and explore the potentiality of data predictions.

When deploying a model, it is important to find a good balance between the functionalities and performances. In fact, to be able to fully appreciate the model, the prediction must be *near real-time.*

One of the simplest way to put an algorithm into production is by creating a web service.
In particular my suggestion is to use a **REST** application which demonstrated to be light and well performing . **Client** and **Server** are independent, this allows us to scale it up fairly easily.

## What about implementation details?

Have you haver heard of *Flask* ? Well it is known for being one of the easiest implementation solutions for server creation. 
Even if it is still considered a great framework, with the birth of **FastAPI** the server realization has become even more simple.
  
FastAPI is a modern, high-performance web framework for **building APIs with Python**.
One of its key points is the speed. It is among the quickest running along with NodeJS.

Imagine to take one of your latest created models. We will take a simple SVM classifier as example:

```python 
import pandas as pd
from sklearn.svm import SVC

df = pd.read_csv('data.csv')

x = df[df.columns.difference(['Target'])

y = df['Target']

classifier = SVC()

classifier.fit(x, y)
```

Considering that is very likely that the environment on which the model will be deployed is different from the one which it was trained, it is necessary to store it in a *pickle* file.

```python 
from sklearn.externals import joblib

joblib.dump(classifier, 'classifier.pkl')
```

Finally we can move on into our application file.
We have simply to instatiate our API object and then create the roots.
In our case we reported :
- *index* which will be shown as server front page 
- *predict*  to fully accomplish the prediction task

```python 
from fastapi import FastAPI

app = FastAPI()

@app.get("/")  
def read_root():  
    return("Hello I am the server")

@app.post("/predict")  
def fetch_preditions(data):  
    query_df = pd.DataFrame(data)
	classifier = joblib.load('classifier.pkl')

	prediction = classifier.predict(query)
	  
    return {"text" : lis(predicition)}
```

# What about Deep Learning models?

When dealing with deep learning models the situation becomes slightly more difficult. In fact they are failry computationally heavier.
The idea is to load previosly obtained checkpoints, but there is the need of modifying the classic generation script to adapt to real-time inference.

It suggested to load the model's checkpoints when starting the application, it will slow the initialization phase but will keep the prediction time very low.
For the sake of simplicity we will report a snippet of code used to perform real-time text generation based on input text.

```python
weights = torch.load("path/to/weights", map_location=torch.device('cpu'))  
model.load_state_dict(weights)

def real_time_inference(text)
source_token_ids = tokenizer.encode(source_keywords, add_special_tokens=False)

results = model.generate(source_token_ids, temperature=0.6)

@app.post("/predict")  
def fetch_preditions(text,):  
    text = text.lower()  
     
  
    prediction = real_time_inference(text)  
  
    print(prediction)  
    return {"Prediction": prediction}
```

The input text is recevied from a *.json* file passed through a **POST** request from the Client.
The following code shows a simple example:

```python
import requests

backend = "http://0.0.0.0:1200/predict"
params = {"text": example}  

x = requests.get(url=backend, params=params)
```


As we have seen, web service model deployment is very easy for simple machine learning models, it can be quite tricky when dealing with really complex deep learning model, but showed to represent the best compromise to put into production your favourite models!
