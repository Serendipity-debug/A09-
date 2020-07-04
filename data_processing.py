import pandas as pd
import numpy as np


def fill_year(data: pd.DataFrame) -> None:
    data['year'] = [2015, 2016, 2017] * (data.shape[0] // 3)


def read_data(base_train_path, base_verify_path,
              report_train_path, report_verify_path,
              money_train_path, money_verify_path, encoding='gbk'):
    base_train = pd.read_csv(base_train_path, encoding=encoding)
    base_verify = pd.read_csv(base_verify_path, encoding=encoding)
    report_train = pd.read_csv(report_train_path, encoding=encoding)
    report_verify = pd.read_csv(report_verify_path, encoding=encoding)
    money_train = pd.read_csv(money_train_path, encoding=encoding)
    money_verify = pd.read_csv(money_verify_path, encoding=encoding)
    # noinspection PyBroadException
    try:
        base_verify.drop('控制人ID', axis=1, inplace=True)
    except:
        pass

    base = pd.concat([base_train, base_verify], axis=0)
    report = pd.concat([report_train, report_verify], axis=0)
    money = pd.concat([money_train, money_verify], axis=0)

    base.dropna(subset=['flag'], inplace=True)

    df = base[['ID', 'flag']]
    report = pd.merge(report, df, on='ID', how='inner')
    money = pd.merge(money, df, on='ID', how='inner')
    fill_year(report)
    fill_year(money)
    year_info = pd.merge(report, money, on=['ID', 'year', 'flag'])
    flag = year_info.pop('flag')
    year_info.insert(year_info.shape[1], 'flag', flag)

    return year_info


def fill_debt(dataset):
    # 填充所有者权益合计
    index = dataset.loc[dataset['所有者权益合计'].isnull()].index
    dataset.iloc[index, 10] = dataset.iloc[index, 3] - dataset.iloc[index, 4]
    # 填充资产总额
    index = dataset.loc[dataset['资产总额'].isnull()].index
    dataset.iloc[index, 3] = dataset.iloc[index, 10] + dataset.iloc[index, 4]
    # 填充负债总额
    index = dataset.loc[dataset['负债总额'].isnull()].index
    dataset.iloc[index, 4] = dataset.iloc[index, 3] - dataset.iloc[index, 10]


def add_debt(dataset):
    # 增加负债率
    debt = dataset.iloc[:, 4] / dataset.iloc[:, 3]
    dataset.insert(5, '负债率', debt)


def fill_finance(dataset):
    for column in ['债权融资', '股权融资', '内部融资和贸易融资', '项目融资和政策融资']:
        dataset.loc[(dataset[column + '额度'].isnull()) & (dataset[column + '成本'] == 0), column + '额度'] = 0
        dataset.loc[(dataset[column + '成本'].isnull()) & (dataset[column + '额度'] == 0), column + '成本'] = 0
        dataset.loc[(dataset[column + '额度'].isnull()) & (dataset[column + '成本'].isnull()),
        column + '额度':column + '成本'] = 0
        finance = dataset.loc[(dataset[column + '额度'] > 0) & (dataset[column + '成本'] > 0),
                  column + '额度':column + '成本'].values
        rate = round((finance[:, 1] / finance[:, 0]).mean(), 2)
        cost = dataset.loc[(dataset[column + '额度'].isnull()) &
                           (dataset[column + '成本'].notnull()), column + '成本'].values
        dataset.loc[(dataset[column + '额度'].isnull()) &
                    (dataset[column + '成本'].notnull()), column + '额度'] = cost / rate
        quota = dataset.loc[(dataset[column + '成本'].isnull()) &
                            (dataset[column + '额度'].notnull()), column + '额度'].values
        dataset.loc[(dataset[column + '成本'].isnull()) &
                    (dataset[column + '额度'].notnull()), column + '成本'] = quota * rate


def fill_profit(dataset):
    # 数据集中，当年净利润小于0的企业，当年纳税总额为0
    dataset.loc[(dataset['净利润'].isnull()) & (dataset['纳税总额'] == 0), '净利润'] = -1
    dataset.loc[(dataset['净利润'].isnull()) & (dataset['纳税总额'] > 0), '净利润'] = 1


def drop_useless(dataset):
    dataset.drop(['债权融资成本', '股权融资成本', '内部融资和贸易融资成本', '项目融资和政策融资成本'], axis=1, inplace=True)


def arrange_feature(dataset: pd.DataFrame) -> pd.DataFrame:
    group_data = dataset.groupby('ID')

    debt = (group_data['负债率'].mean() >= 0.85).astype('int8')
    loss = group_data['净利润'].apply(lambda x: (x < 0).all()).astype('int8')
    quota = group_data[['债权融资额度', '股权融资额度', '内部融资和贸易融资额度', '项目融资和政策融资额度']].sum()
    if 'flag' in dataset.columns:
        flag = group_data['flag'].apply(lambda x: (x == 1).all()).astype('int8')
        return pd.concat([loss, debt, quota, flag], axis=1)
    else:
        return pd.concat([loss, debt, quota], axis=1)


def train_data_processing(base_train_path, base_verify_path,
                          report_train_path, report_verify_path,
                          money_train_path, money_verify_path,
                          encoding='gbk'):
    dataset = read_data(base_train_path, base_verify_path,
                        report_train_path, report_verify_path,
                        money_train_path, money_verify_path, encoding=encoding)
    fill_debt(dataset)
    fill_finance(dataset)
    fill_profit(dataset)
    add_debt(dataset)
    drop_useless(dataset)
    return arrange_feature(dataset)


def test_data_processing(dataset):
    add_debt(dataset)
    drop_useless(dataset)
    return arrange_feature(dataset)
