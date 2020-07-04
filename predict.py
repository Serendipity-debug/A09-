import pandas as pd
import pickle
from data_processing import test_data_processing
from time import process_time


# import xgboost as xgb


def read_data(path_year, path_money, encoding='gbk') -> pd.DataFrame:
    year_data = pd.read_csv(path_year, encoding=encoding)
    money_data = pd.read_csv(path_money, encoding=encoding)
    return pd.merge(year_data, money_data, on=['ID', 'year'])


def predict(path_year, path_money, out_path, encoding='gbk', save=True):
    data = read_data(path_year, path_money, encoding=encoding)
    data = test_data_processing(data)
    x = data.values
    # 加载模型
    loaded_model = pickle.load(open('./xgb.pickle.dat', 'rb'))
    pred = loaded_model.predict(x)
    pred = [round(value) for value in pred]
    df = pd.DataFrame(pred, index=data.index, columns=['pred'])
    if save:
        df.to_csv(out_path)
    else:
        return df


if __name__ == '__main__':
    start = process_time()

    year = './predict-data/year_report_test_sum.csv'
    money = './predict-data/money_report_test_sum.csv'
    out = './predict-data/result.csv'

    predict(year, money, out, encoding='utf-8')

    end = process_time()
    print('time:{time}s'.format(time=end - start))
