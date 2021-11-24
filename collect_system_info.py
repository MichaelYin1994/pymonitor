#!/opt/conda/bin/python
# -*- coding: utf-8 -*-

# Created on 202111221842
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

'''
搜集系统级别的CPU与GPU信息，并写入本地SQLite3数据库。

References:
----------
[1] https://www.cnblogs.com/TonvyLeeBlogs/articles/13800115.html
[2] https://psutil.readthedocs.io/en/latest/#psutil.cpu_times
[3] https://www.thepythoncode.com/article/get-hardware-system-information-python
[4] https://stackoverflow.com/questions/33076617/how-to-validate-time-format
'''

import os
import sqlite3
import time
from datetime import datetime
from functools import wraps

import numpy as np
import pandas as pd
import psutil
import pynvml

# 全局环境与参数初始化
COLLECT_SYS_INFO_EVERY_K_SECONDS = 7
DATABASE_DIR = 'system_logs_database'
DATABASE_NAME = 'sys_resource_usage_logs.db'
TABLE_NAME = 'sys_resource_logs_table'


def add_prefix(info_dict, prefix=None):
    '''为字典key添加标识符'''
    if prefix is None:
        return info_dict
    if not isinstance(info_dict, dict):
        raise ValueError('Invalid input info_dict !')

    tmp_dict = {}
    for key in info_dict.keys():
        tmp_dict['{}_{}'.format(prefix, key)] = info_dict[key]

    return tmp_dict


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


def collect_cpu_info():
    '''采集server端的CPU信息，并返回cpu信息的字典'''
    cpu_stats_dict = {}

    # 获取cpu的频率情况
    cpu_freq = psutil.cpu_freq()
    cpu_stats_dict['min_freq'] = cpu_freq.min
    cpu_stats_dict['max_freq'] = cpu_freq.max
    cpu_stats_dict['current_freq'] = cpu_freq.current

    # 获取cpu的核心情况
    cpu_stats_dict['n_logical_cores'] = psutil.cpu_count(logical=True)
    cpu_stats_dict['n_physical_cores'] = psutil.cpu_count(logical=False)

    # 获取cpu的使用情况
    cpu_usage_stats = psutil.cpu_percent(percpu=True, interval=1)
    cpu_stats_dict['usage-25'] = np.quantile(cpu_usage_stats, 0.25)
    cpu_stats_dict['usage-50'] = np.quantile(cpu_usage_stats, 0.5)
    cpu_stats_dict['usage-75'] = np.quantile(cpu_usage_stats, 0.75)
    cpu_stats_dict['usage-mean'] = np.mean(cpu_usage_stats)

    # 为key添加标识符
    cpu_stats_dict = add_prefix(cpu_stats_dict, 'cpu')

    return cpu_stats_dict


def collect_network_info():
    '''采集server端的Network信息，并返回Network信息的字典'''
    network_stats_dict = {}
    net_io = psutil.net_io_counters()

    # network收发量
    network_stats_dict['mb-sent'] = net_io.bytes_sent / 1024**2
    network_stats_dict['mb-receive'] = net_io.bytes_recv / 1024**2

    # 为key添加标识符
    network_stats_dict = add_prefix(network_stats_dict, 'network')

    return network_stats_dict


def collect_memory_info():
    '''采集server端的Memory信息，并返回Memory信息的字典'''
    memory_stats_dict = {}
    memory_info = psutil.virtual_memory()
    swap_info = psutil.swap_memory()

    # 内存信息保存
    memory_stats_dict['total_system_memory'] = memory_info.total / 1024**2
    memory_stats_dict['available_system_memory'] = memory_info.available / 1024**2
    memory_stats_dict['used_system_memory'] = memory_info.used / 1024**2
    memory_stats_dict['free_system_memory'] = memory_info.free / 1024**2

    # swap信息保存
    memory_stats_dict['used_system_swap'] = swap_info.total / 1024**2
    memory_stats_dict['used_system_swap'] = swap_info.used / 1024**2
    memory_stats_dict['free_system_swap'] = swap_info.free / 1024**2

    return memory_stats_dict


def collect_gpu_info(gpu_id=None):
    '''搜集指定ID的GPU的使用率、显存使用率'''
    if gpu_id is None:
        gpu_id = 0

    gpu_stats_dict = {}
    gpu_stats_dict['gpu_id'] = gpu_id
    pynvml.nvmlInit()

    # 获取gpu实例
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    gpu_use = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)

    # 获取gpu使用率
    gpu_stats_dict['core_use'] = gpu_use.gpu / 100

    # 获取gpu温度
    gpu_stats_dict['temperature'] = pynvml.nvmlDeviceGetTemperature(
        gpu_handle, pynvml.NVML_TEMPERATURE_GPU
    )

    # 获取gpu的显存占用
    gpu_memory_usage = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    gpu_stats_dict['memory_total'] = gpu_memory_usage.total / 1024**2
    gpu_stats_dict['memory_used'] = gpu_memory_usage.used / 1024**2
    gpu_stats_dict['memory_free'] = gpu_memory_usage.free / 1024**2

    # 获取gpu的功率使用
    gpu_stats_dict['tdp'] = pynvml.nvmlDeviceGetEnforcedPowerLimit(gpu_handle) / 1000
    gpu_stats_dict['power_use'] = pynvml.nvmlDeviceGetPowerUsage(gpu_handle) / 1000

    # 获取gpu的风扇使用率
    gpu_stats_dict['fan_use'] = pynvml.nvmlDeviceGetFanSpeed(gpu_handle)

    pynvml.nvmlShutdown()

    # 为key添加标识符
    gpu_stats_dict = add_prefix(gpu_stats_dict, 'gpu')

    return gpu_stats_dict


@timefn
def collect_system_hardware_level_info(**kwargs):
    '''全面搜集系统层面的资源消耗情况'''
    gpu_id = kwargs.pop('gpu_id', 0)
    system_stats_dict = {
        'collect_datetime': str(datetime.now())
    }

    # 系统内存信息
    system_stats_dict = {
        **system_stats_dict, **collect_memory_info()
    }

    # 系统CPU信息
    system_stats_dict = {
        **system_stats_dict, **collect_cpu_info()
    }

    # 系统GPU信息
    system_stats_dict = {
        **system_stats_dict, **collect_gpu_info(gpu_id)
    }

    # 系统Network信息
    system_stats_dict = {
        **system_stats_dict, **collect_network_info()
    }

    return system_stats_dict


def initial_local_environment():
    '''初始化数据库本地环境'''
    if 'system_logs_database' not in os.listdir():
        os.mkdir('system_logs_database')


def query_system_info(start_time, end_time):
    '''查询sys_info.db中start_time到end_time时间范围的系统基础信息'''
    check_datetime_format(start_time)
    check_datetime_format(end_time)
    database_name_tmp = os.path.join(DATABASE_DIR, DATABASE_NAME)

    with sqlite3.connect(database_name_tmp) as conn:
        cursor = conn.cursor()

        # 提取表头
        cursor.execute(
            'SELECT * FROM {} LIMIT 1'.format(TABLE_NAME)
        )
        col_name_list = [item[0] for item in cursor.description]

        # 组装SQL Query语句
        query_res = "SELECT * FROM {} WHERE {}.collect_datetime BETWEEN '{}' AND '{}';".format(
            TABLE_NAME, TABLE_NAME, start_time, end_time
        )

        # 执行查询语句
        sys_logs = cursor.execute(query_res)

        query_res = [item for item in list(sys_logs)]
        query_df = pd.DataFrame(query_res, columns=col_name_list)
        query_df.drop(['index'], axis=1, inplace=True)

    return query_df


if __name__ == '__main__':
    # 初始化本地环境
    # **********************
    initial_local_environment()
    database_name = os.path.join(DATABASE_DIR, DATABASE_NAME)

    system_info_dict = collect_system_hardware_level_info()
    system_info_dict_keys = list(system_info_dict.keys())

    # 连接SQLite数据库，并写入系统信息
    # **********************
    with sqlite3.connect(database_name) as conn:

        while(True):
            system_info_dict = collect_system_hardware_level_info()

            system_info_df = pd.DataFrame(system_info_dict, index=[0])
            system_info_df['collect_datetime'] = pd.to_datetime(
                system_info_df['collect_datetime']
            )

            system_info_df.to_sql(
                TABLE_NAME, con=conn, if_exists='append'
            )

            time.sleep(COLLECT_SYS_INFO_EVERY_K_SECONDS)

        cursor.close()
        conn.commit()
