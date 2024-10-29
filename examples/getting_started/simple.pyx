# cython: language_level=3

def add(x: int, y: int):
    return x + y


cdef float linear(slope: float, x: float, b: float):
    return slope * x + b

def two_linear(slope: float, x: float, b: float):
    cdef:
        float r1 = linear(slope, x, b)
        float r2 = linear(-slope, x, b)
        float result = r1 + 2 * r2

    return result
