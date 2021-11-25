
from utils.io_utils import query_system_info

if __name__ == '__main__':
    tmp_df = query_system_info(
        '2021-11-24 12:00:00', '2021-11-24 12:10:00'
    )
