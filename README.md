# Prediction APP of Workloads 
If you want to predict how many employers you might need next year, just input previous data and influencial elements into this app, it will come out the workoads next year.
---------------------------
This APP provides a prediction website which uses Sklearn regression model to predict HR's workloads next year. Most of the time developers use Sklearn simply to build a model or classifier. They barely build a website to provide APIs for users. With powerful Werkzeug WSGI tool box and Jinja2 templates, Flask is imported in this project to enable more users to implement the model.

CLONE NOW AND GET STARTED!
### Watch the DEMO
Demo gif animation is here  ![image](https://raw.githubusercontent.com/anqi666/PredictionAPP/main/demoapp.gif )   
### Project Structure
---------------------------

- CSVdata 
    - NextYearData1.csv
    - NextYearData2.csv
    - PastYearData1.csv
    - PastYearData2.csv
- app.py 
- static 
    - corr.jpg
    - rmse.png
    - style.css
    - train_predict.jpg
    - uploads
        - DATA2020.csv 
        - INPUTS_2021.csv
        - test.joblib
- templates 
    - predic.html
    - train.html
    - upload.html    
### Dependencies
---------------------------
- scikit-learn
- Flask
- pandas
- numpy
- matplotlib

If you don't have Flask, 

      $ pip install flask
### Running API
---------------------------
```
python app.py 
