#!/opt/conda/bin/python
# -*- coding: utf-8 -*-

# Created on 202111241515
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

'''
用于连接数据库、环境初始化的IO工具。
'''

import sys

sys.path.append('..')

import os
import sqlite3
import time
from datetime import datetime
from functools import wraps

import pandas as pd
from config import Configs


def check_datetime_format(datetime_str):
    '''检查时间字符串是否合法'''
    if not isinstance(datetime_str, str):
        raise ValueError('Invalid input datetime type !')

    try:
        datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        raise ValueError("Incorrect data format, should be YYYY-MM-DD %H:%M:%S")


def timefn(func):
    '''用于函数运行时间测量的装饰器'''
    @wraps(func)
    def measure_time(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(
            '{} TIME COST: {} took: {:.5f} seconds.'.format(
                str(datetime.now())[:-4], func.__name__, end - start
            )
        )
        return result
    return measure_time

def initial_local_environment():
    '''初始化数据库本地环境'''
    if 'system_logs_database' not in os.listdir():
        os.mkdir('system_logs_database')


def query_system_info(start_time, end_time, **kwargs):
    '''查询sys_info.db中start_time到end_time时间范围的系统基础信息'''
    check_datetime_format(start_time)
    check_datetime_format(end_time)

    database_dir = kwargs.pop('database_dir', Configs.DATABASE_DIR)
    database_name = kwargs.pop('database_name', Configs.DATABASE_NAME)
    table_name = kwargs.pop('database_name', Configs.TABLE_NAME)

    database_name_tmp = os.path.join(database_dir, database_name)

    with sqlite3.connect(database_name_tmp) as conn:
        cursor = conn.cursor()

        # 提取表头
        cursor.execute(
            'SELECT * FROM {} LIMIT 1'.format(table_name)
        )
        col_name_list = [item[0] for item in cursor.description]

        # 组装SQL Query语句
        query_res = "SELECT * FROM {} WHERE {}.collect_datetime BETWEEN '{}' AND '{}';".format(
            table_name, table_name, start_time, end_time
        )

        # 执行查询语句
        sys_logs = cursor.execute(query_res)

        query_res = [item for item in list(sys_logs)]
        query_df = pd.DataFrame(query_res, columns=col_name_list)
        query_df.drop(['index'], axis=1, inplace=True)

    return query_df
