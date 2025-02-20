import torch


class Nil:
    def __repr__(self):
        return "nil"

    def __bool__(self):
        return False


nil = Nil()


def r(num, d=0):
    return round(num) if d == 0 else round(num, d)
