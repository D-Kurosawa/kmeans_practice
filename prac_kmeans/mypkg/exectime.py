"""Measure execution time"""
import time

from dateutil.relativedelta import relativedelta


def app_time(func):
    """Decorator to application execution time"""

    def wrapper(*args, **kwargs):
        s = time.time()
        print(f"\n<{'-' * 78}>")
        print(f'>>> Console Application Start\n')

        result = func(*args, **kwargs)

        e = time.time() - s
        normal = relativedelta(seconds=e).normalized()
        print(f"\n{'=' * 80}")
        print(f"Elapsed Time : {e:.10f} [sec]")
        print(f"             : {normal.days} Day \t"
              f"{normal.hours:02}:{normal.minutes:02}:"
              f"{normal.seconds:02}.{normal.microseconds}")
        print(f"{'=' * 80}")
        print(f'\n>>> Console Application End')
        print(f"<{'-' * 78}>\n")
        return result

    return wrapper


def func_time(func):
    """Decorator to function execution time"""

    def wrapper(*args, **kwargs):
        s = time.time()

        result = func(*args, **kwargs)

        e = time.time() - s
        print(f"{'-' * 100}")
        print(f"function name : {func.__qualname__} : {e:.10f} [sec]")
        print(f"{'-' * 100}")
        return result

    return wrapper


if __name__ == '__main__':
    pass
