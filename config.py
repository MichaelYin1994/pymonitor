#!/opt/conda/bin/python
# -*- coding: utf-8 -*-

# Created on 202111241518
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

'''
工具参数配置类。
'''

class Configs():
    '''参数搜集配置文件'''
    DATABASE_DIR = 'system_logs_database'
    DATABASE_NAME = 'sys_resource_usage_logs.db'
    TABLE_NAME = 'sys_resource_logs_table'

    COLLECT_SYSTEM_INFO_EVERY_K_SECONDS = 7
