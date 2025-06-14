from pandas import to_datetime
import time


def unix_to_datetime(ts, unit="s"):
    return to_datetime(ts, unit=unit)


def str_to_datetime(ts, unit="s"):
    return to_datetime(ts).round(unit).tz_localize(None)


def rate_limit(interval):
    def decorator(func):
        last_called = [0]

        def wrapper(*args, **kwargs):
            now = time.time()
            if now - last_called[0] >= interval:
                last_called[0] = now
                return func(*args, **kwargs)

        return wrapper

    return decorator
