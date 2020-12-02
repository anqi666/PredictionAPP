from flask import Flask, render_template, request, flash, redirect, url_for, make_response
from flask_bootstrap import Bootstrap
import os

import joblib

import numpy as np
import pandas as pd
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from skimage.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import datetime
from datetime import timedelta

matplotlib.use('Agg')
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)
bootstrap = Bootstrap(app)
app.config['SECRET_KEY'] = os.urandom(24)
basedir = os.path.abspath(os.path.dirname(__file__))
uploadDir = os.path.join(basedir, 'static/uploads')
uploadPath1 = os.path.join(basedir, 'static/uploads/DATA2020.csv')
uploadPath2 = os.path.join(basedir, 'static/uploads/INPUTS_2021.csv')
filename_model = os.path.join(basedir, 'static/uploads/test.joblib')


# uploadPath1 = '/Users/angela/Documents/CSVread/static/uploads/DATA2020.csv'
# uploadPath2 = '/Users/angela/Documents/CSVread/static/uploads/INPUTS_2021.csv'
# filename_model = '/Users/angela/Documents/CSVread/static/uploads/test.joblib'


# ===========================读取上传文件========================================
@app.route('/', methods=['POST', 'GET'])
def process():
    if request.method == 'POST':
        f1 = request.files.get('fileupload1')
        f2 = request.files.get('fileupload')
        if not os.path.exists(uploadDir):
            os.makedirs(uploadDir)

        if f1:
            f1.save(uploadPath1)
            flash('Upload Load Successful!', 'success')
            csv_data = open(uploadPath1, 'rb').read()
            response = make_response(csv_data)
            response.headers['Content-Type'] = 'csv'
            return response
        if f2:
            f2.save(uploadPath2)
            flash('Upload Load Successful!', 'success')
            csv_data = open(uploadPath2, 'rb').read()
            response = make_response(csv_data)
            response.headers['Content-Type'] = 'csv'
            return response

        else:
            flash('No File Selected.', 'danger')
        return redirect(url_for('process'))
    return render_template('upload.html')


# ===========================训练回归模型========================================
@app.route('/train', methods=['POST'])
def train():
    if request.method == 'POST':
        hist_data = pd.read_csv(uploadPath1)
        X = np.array(hist_data.values[:, 1:7].tolist())
        Y = np.array(hist_data.values[:, 7].tolist())
        validation_size = 0.2

        X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

        # print(dataset.corr(method='pearson'))
        data_corr = pd.DataFrame(hist_data.corr(method='pearson'))
        # print(data_corr)
        dc = data_corr["Workload_Amount"]
        corr_sort = dc.sort_values(ascending=False)[1:7]

        # print(type(corr_sort.values))
        Feature = {'FeatureName': corr_sort.index, 'FeatureCorrelationValue': np.around(corr_sort.values, decimals=3)}
        Feature_pd = pd.DataFrame(Feature)
        FT = Feature_pd.values
        Feature_pd.index = corr_sort.index
        #
        Feature_pd.plot(kind='bar')
        plt.axhline(0)
        plt.tight_layout()

        plt.savefig(os.path.join(basedir, 'static/corr.jpg'))
        plt.close()
        # 　SKlearn Pipeline=========================================================
        pipe = Pipeline([('Scaler', StandardScaler()), ('LR', LinearRegression())])
        pipe.fit(X_train, Y_train)

        sum = 'Workload = '
        for i, j in zip(pipe['LR'].coef_[0:6], hist_data.head(0).iloc[:, 1:7]):
            sum += '(%.2f) * %s  + ' % (i, j)
        formula_str = sum[:-2]

        kfold = KFold(n_splits=10, random_state=7)
        cv_result = cross_val_score(pipe, X_train, Y_train, cv=kfold, scoring='neg_root_mean_squared_error')
        #     results.append(cv_result)
        me = np.around(cv_result.mean(), decimals=3)
        std = np.around(cv_result.std(), decimals=3)

        Y_prediction1 = pipe.predict(X_train)
        # plt.scatter(range(17), Y_prediction1, color='red')
        # plt.scatter(range(17), Y_train, color='blue')

        fig = plt.figure()
        start = datetime.datetime(2019, 1, 1)
        stop = datetime.datetime(2020, 4, 1)
        delta = datetime.timedelta(weeks=4)
        dates = mpl.dates.drange(start, stop, delta)  #
        y1 = Y_train
        y2 = Y_prediction1
        ax = plt.gca()
        ax.plot_date(dates, y1, linestyle='-', marker='s')
        ax.plot_date(dates, y2, linestyle='-', marker='o', color='red')

        date_format = mpl.dates.DateFormatter('%Y-%m')
        # 只显示年月
        ax.xaxis.set_major_formatter(date_format, )
        fig.autofmt_xdate()
        # 开启自适应
        plt.legend(['Prediction value', 'Real value'])

        plt.title("Prediction in training set")
        plt.xlabel("Month")
        plt.ylabel("Workloads")
        # plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(basedir, 'static/train_predict.jpg'))
        Y_prediction = pipe.predict(X_validation)
        Y_ = np.floor(Y_prediction)

        errors = (np.sqrt(mean_squared_error(Y_, Y_validation))) / Y_validation.mean()
        acc = 100 - np.around(errors, decimals=4) * 100
        joblib.dump(pipe, filename_model)

        return render_template('train.html', Feature=FT.tolist(), acc=acc, mean=me, std=std, formula_str=formula_str)
    # return render_template('train.html', train_result=X)


# ===========================预测下一年工作量========================================
@app.route('/predict', methods=['POST'])
def predict():
    LR_model = open(filename_model, 'rb')
    pipe = joblib.load(LR_model)
    df = pd.read_csv(uploadPath1)
    val = df.values[:, 1:7].tolist()
    X_np = np.array(val)
    Y_np = np.floor(pipe.predict(X_np))
    YY = Y_np.tolist()
    names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    nvDict = dict((name, value) for name, value in zip(names, YY))
    return render_template('predic.html', prediction=nvDict)


if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0')
