import math


class DualNumber:
    def __init__(self, value, dvalue):
        self.value = value
        self.dvalue = dvalue

    # other is also DualNumber
    def __mul__(self, other):
        return DualNumber(self.value * other.value, self.dvalue * other.value + self.value * other.dvalue)

    def __add__(self, other):
        return DualNumber(self.value + other.value, self.dvalue + other.dvalue)

    def __truediv__(self, other):
        return DualNumber(self.value / other.value, (self.dvalue * other.value - self.value * other.dvalue) / other.value**2)

    def __sub__(self, other):
        return DualNumber(self.value - other.value, self.dvalue - other.dvalue)

    def __pow__(self, other):
        return DualNumber(self.value ** other.dvalue, self.value**other.value * (other.dvalue * math.log(self.value) + other.value * (1/self.value) * self.dvalue))


def sin(x):
    return DualNumber(math.sin(x.value), math.cos(x.value) * x.dvalue)


def cos(x):
    return DualNumber(math.cos(x.value), -math.sin(x.value) * x.dvalue)


def tan(x):
    return DualNumber(math.tan(x.value), 1/math.cos(x.value)**2 * x.dvalue)


def exp(x):
    return DualNumber(math.exp(x.value), math.exp(x.value) * x.dvalue)


# Test
if __name__ == '__main__':
    assert cos(DualNumber(0, 1)).value == 1
    assert tan(DualNumber(0, 1)).value == 0
    assert exp(DualNumber(0, 1)).value == 1


def z(x, y):
    return x * y + sin(x)

z_value = z(DualNumber(0.5, 1), DualNumber(4.2, 0)).value
z_dvalue_x = z(DualNumber(0.5, 1), DualNumber(4.2, 0)).dvalue
z_dvalue_y = z(DualNumber(0.5, 0), DualNumber(4.2, 1)).dvalue
print('z is ', z_value)
print('z_x is', z_dvalue_x)
print('z_y is', z_dvalue_y)