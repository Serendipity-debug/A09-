import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from data_processing import train_data_processing
import pickle
from time import process_time

if __name__ == '__main__':
    start = process_time()
    data = train_data_processing(base_train_path='./data/raw_data/base_train_sum.csv',
                                 base_verify_path='./data/raw_data/base_verify1.csv',
                                 report_train_path='./data/raw_data/year_report_train_sum.csv',
                                 report_verify_path='./data/raw_data/year_report_verify1.csv',
                                 money_train_path='./data/raw_data/money_report_train_sum.csv',
                                 money_verify_path='./data/raw_data/money_information_verify1.csv')

    x, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    model = xgb.XGBClassifier()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    pred = [round(value) for value in pred]
    print(model.__class__.__name__)
    print('accuracy_score:{score}%'.format(score=accuracy_score(y_test, pred) * 100))
    print('precision_score:{score}%'.format(score=precision_score(y_test, pred) * 100))
    print('recall_score:{score}%'.format(score=recall_score(y_test, pred) * 100))
    print('f1_score:{score}%'.format(score=f1_score(y_test, pred) * 100))

    pickle.dump(model, open('xgb.pickle.dat', 'wb'))
    end = process_time()
    print(end - start)
