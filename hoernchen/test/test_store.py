import hoernchen
from hoernchen import col, table, Store
from dataclasses import dataclass


@dataclass
class A:
    a: str
    b: int

@dataclass
class B:
    a: str
    b: int


def test_insert_and_get():

    s = Store(":memory:")
    a1 = A("a", 3)

    s.insert(a1)

    a2 = s.get(A, b=3)[0]

    assert a1 == a2


def test_select1():

    s = Store(":memory:")

    for i in range(10):
        a = A("a", i)
        s.insert(a)

    rs = s.select(table(A).where(col("b") > 7))
    assert len(rs) == 2


def test_join1():

    s = Store(":memory:")

    for i in range(10):
        a = A("a", i)
        s.insert(a)

    for i in range(3):
        a = A("b", i)
        s.insert(a)
    s.insert(B('a', 1))

    rs = s.select(table(A).join(table(B), on='a'))
    assert len(rs) == 10

def test_join2():

    s = Store(":memory:")

    for i in range(10):
        a = A("a", i)
        s.insert(a)

    for i in range(3):
        a = A("b", i)
        s.insert(a)
    s.insert(B('a', 1))

    rs = s.select(table(A).join(table(B), on='a', how='left').where(col('a') == None))
    assert len(rs) == 3

def test_join2():

    s = Store(":memory:")

    for i in range(10):
        a = A("a", i)
        s.insert(a)

    for i in range(3):
        a = A("b", i)
        s.insert(a)
    s.insert(B('a', 1))

    rs = s.select(table(A).order_by(col("b")).join(table(B), on='a', how='left').where(col('a') == None))

    assert rs[0] == A("b", 0)
    assert rs[1] == A("b", 1)
    assert rs[2] == A("b", 2)

    rs = s.select(table(A).order_by(col("b"), "desc").join(table(B), on='a', how='left').where(col('a') == None))

    assert rs[0] == A("b", 2)
    assert rs[1] == A("b", 1)
    assert rs[2] == A("b", 0)


def test_limit1():

    s = Store(":memory:")

    for i in range(10):
        a = A("a", i)
        s.insert(a)

    for i in range(3):
        a = A("b", i)
        s.insert(a)
    s.insert(B('a', 1))

    rs = s.select(table(A).order_by(col("b")).join(table(B), on='a', how='left').where(col('a') == None), limit=1)

    assert len(rs) == 1
    assert rs[0] == A("b", 0)

@dataclass
class CBoolean:
    a: str
    b: int
    c: bool


def test_boolean():
    s = Store(":memory:")
    o1 = CBoolean('hello', 3, True)
    o2 = CBoolean('World', 4, False)
    s.insert([o1, o2])


    s1 = s.get(CBoolean, b=3)[0]
    s2 = s.get(CBoolean, b=4)[0]

    assert o1 == s1
    assert o2 == s2

    assert isinstance(s1.c, bool)


def test_update():
    s = Store(":memory:")
    o1 = CBoolean('hello', 3, True)
    o2 = CBoolean('World', 4, False)
    s.insert(o1)
    s.insert(o2, update=True)

    s1 = s.get(CBoolean, b=3)[0]
    s2 = s.get(CBoolean, b=4)[0]

    assert o1 == s1
    assert o2 == s2

    assert isinstance(s1.c, bool)


@dataclass
class D:
    a: int
    b: int


def test_get_order_by():

    s = Store(":memory:")
    l = []
    for i in range(10):
        d = D(i, 10-i)
        l.append(d)
        s.insert(d)

    assert l == s.get_where(D, order_by='a')
    assert l[::-1] == s.get_where(D, order_by='b')
    assert l[::-1] == s.get_where(D, order_by='a', reverse=True)
    assert l == s.get_where(D, order_by='b', reverse=True)


def test_get_order_by_limit():

    s = Store(":memory:")
    l = []
    for i in range(10):
        d = D(i, 10-i)
        l.append(d)
        s.insert(d)

    assert l[:3] == s.get_where(D, order_by='a', limit=3)
    assert l[::-1][:3] == s.get_where(D, order_by='b', limit=3)
    assert l[::-1][:3] == s.get_where(D, order_by='a', reverse=True, limit=3)
    assert l[:3] == s.get_where(D, order_by='b', reverse=True, limit=3)

