from __future__ import absolute_import
from __future__ import print_function


def is_ascending(lst):
    for i in range(1, len(lst)):
        if lst[i-1] > lst[i]:
            return 0  # 如果存在非升序的情况，返回0
    return 1  # 列表是升序的情况下返回1


