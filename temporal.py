from dateutil import parser
from pandas import to_datetime
from datetime import datetime


def unix_to_datetime(ts, unit="s"):
    return to_datetime(ts, unit=unit)


def str_to_datetime(ts):
    return parser.isoparse(ts).replace(microsecond=0)


def dt_to_str(ts: str) -> str:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime(
        "%d-%m-%y %H:%M:%S"
    )


# def rate_limit(interval):
#     def decorator(func):
#         last_called = [0]
#
#         def wrapper(*args, **kwargs):
#             now = time.time()
#             if now - last_called[0] >= interval:
#                 last_called[0] = now
#                 return func(*args, **kwargs)
#
#         return wrapper
#
#     return decorator
