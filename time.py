from pandas import to_datetime


def unix_to_datetime(ts, unit="s"):
    return to_datetime(ts, unit=unit)


def str_to_datetime(ts, unit="s"):
    return to_datetime(ts).round(unit).tz_localize(None)
