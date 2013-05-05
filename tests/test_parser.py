from pygpu.parser import Compiler

OPERATORS = ['+', '-', '*', '/', '%', '//', '**', 'or', 'and', 'is', 'is not',
             '<', '<=', '>', '>=', '!=', '==', '|', '^', '&', '<<', '>>']
PREFIXES = ['+', '-', 'not ', '~']


def test_operators():
    for op in OPERATORS:
        yield operators, op

def operators(op):
    c = Compiler()
    s = c.parse("a %s 1" % (op,))
    assert len(s) == 1
    t = s[0]
    assert t.id == op
    assert t.first.id == '(name)'
    assert t.first.value == 'a'
    assert t.second.id == '(number)'
    assert t.second.value == '1'

def test_prefixes():
    for p in PREFIXES:
        yield prefixes, p

def prefixes(p):
    c = Compiler()
    s = c.parse("%sb" % (p,))
    p = p.strip() # remove the space after the not
    assert len(s) == 1
    t = s[0]
    assert t.id == p
    assert t.first.id == '(name)'
    assert t.first.value == 'b'
    assert t.second is None

def test_if_else():
    c = Compiler()
    s = c.parse("a if b else c")
    assert len(s) == 1
    t = s[0]
    assert t.id == 'if'
    assert t.first.id == '(name)'
    assert t.first.value == 'a'
    assert t.second.id == '(name)'
    assert t.second.value == 'b'
    assert t.third.id == '(name)'
    assert t.third.value == 'c'

def test_grouping():
    c = Compiler()
    s = c.parse("(a)")
    assert len(s) == 1
    t = s[0]
    assert t.id == '('
    assert t.first.id == '(name)'
    assert t.first.value == 'a'

    exc = None
    try:
        s = c.parse("(a,)")
    except Exception as e:
        exc = e
    assert isinstance(exc, SyntaxError)

    exc = None
    try:
        c.parse("(a, b)")
    except Exception as e:
        exc = e
    assert isinstance(exc, SyntaxError)

def test_index():
    c = Compiler()
    s = c.parse("a[1]")
    assert len(s) == 1
    t = s[0]
    assert t.id == '['
    assert t.first.id == '(name)'
    assert t.first.value == 'a'
    assert len(t.second) == 1
    assert t.second[0].id == '(number)'
    assert t.second[0].value == '1'

    s = c.parse("a[1, 2, 3]")
    assert len(s) == 1
    t = s[0]
    assert t.id == '['
    assert t.first.id == '(name)'
    assert t.first.value == 'a'
    assert len(t.second) == 3
    assert t.second[0].id == '(number)'
    assert t.second[0].value == '1'
    assert t.second[1].id == '(number)'
    assert t.second[1].value == '2'
    assert t.second[2].id == '(number)'
    assert t.second[2].value == '3'

    s = c.parse("a.i[1, 2]")
    assert len(s) == 1
    t = s[0]
    assert t.id == '['
    assert t.first.id == '.'
    assert t.first.first.id == '(name)'
    assert t.first.first.value == 'a'
    assert t.first.second.id == '(name)'
    assert t.first.second.value == 'i'
    assert len(t.second) == 2
    assert t.second[0].id == '(number)'
    assert t.second[0].value == '1'
    assert t.second[1].id == '(number)'
    assert t.second[1].value == '2'

    exc = None
    try:
        c.parse('a[:]')
    except Exception as e:
        exc = e
    assert isinstance(e, SyntaxError)

    exc = None
    try:
        c.parse('a[...]')
    except Exception as e:
        exc = e
    assert isinstance(e, SyntaxError)

    exc = None
    try:
        c.parse('2[1]')
    except Exception as e:
        exc = e
    assert isinstance(e, SyntaxError)


def test_call():
    c = Compiler()
    s = c.parse("pow(a, b)")
    assert len(s) == 1
    t = s[0]
    assert t.id == '('
    assert t.first.id == '(name)'
    assert t.first.value == 'pow'
    assert len(t.second) == 2
    assert t.second[0].id == '(name)'
    assert t.second[0].value == 'a'
    assert t.second[1].id == '(name)'
    assert t.second[1].value == 'b'

    # we don't check the "f(a=b)" case since it actually 
    # passes the parser and is caught at the check level.

    exc = None
    try:
        c.parse('f(*args)')
    except Exception as e:
        exc = e
    assert isinstance(e, SyntaxError)

    exc = None
    try:
        c.parse('f(a, **kwargs)')
    except Exception as e:
        exc = e
    assert isinstance(e, SyntaxError)


def test_multi_statements():
    c = Compiler()
    s = c.parse("a = b, b = a")
    assert len(s) == 2
    
    exc = None
    try:
        c.parse('a = b b = a')
    except Exception as e:
        exc = e
    assert isinstance(e, SyntaxError)
    
    try:
        c.parse('a = b , , a')
    except Exception as e:
        exc = e
    assert isinstance(e, SyntaxError)

