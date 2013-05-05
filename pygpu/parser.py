import re
import tokenize
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

def parse(program):
    global token, next
    next = _tokenize(program).next
    token = next()
    statements = []
    while True:
        statements.append(_parse(0))
        if token.id != ",":
            break;
        advance(",")
    assert token.id == "(end)"
    return statements

def _tokenize(program):
    for t in tokenize.generate_tokens(StringIO(program).next):
        s = None
        if t[0] == tokenize.NUMBER:
            s = symbol_table['(number)']()
            s.value = t[1]
        elif t[0] == tokenize.NAME:
            sym = symbol_table.get(t[1])
            if sym:
                s = sym()
            else:
                s = symbol_table['(name)']()
                s.value = t[1]
        elif t[0] == tokenize.ENDMARKER:
            s = symbol_table['(end)']()
        elif t[0] == tokenize.OP:
            try:
                s = symbol_table[t[1]]()
            except KeyError:
                raise SyntaxError('Unknown operator: ' + t[1])
        yield s

token = None

def _parse(rbp):
    global token
    t = token
    token = next()
    left = t.nud()
    while rbp < token.lbp:
        t = token
        token = next()
        left = t.led(left)
    return left

class symbol_base(object):
    id = None
    value = None
    first = second = third = None

    def nud(self):
        raise SyntaxError('nud')

    def led(self, left):
        raise SyntaxError('led')

    def __repr__(self):
        if self.id == "(name)" or self.id == "(number)":
            return "(%s %s)" % (self.id[1:-1], self.value)
        else:
            return "(" + " ".join(map(repr, filter(None, [self.id, self.first, self.second, self.third]))) + ")"

symbol_table = {}

def symbol(id, bp=0):
    try:
        s = symbol_table[id]
    except KeyError:
        class s(symbol_base):
            pass
        s.__name__ = "symbol-" + id
        s.id = id
        s.lbp = bp
        symbol_table[id] = s
    else:
        s.lbp = max(bp, s.lbp)
    return s

def infix(id, bp):
    def led(self, left):
        self.first = left
        self.second = _parse(bp)
        return self
    symbol(id, bp).led = led

def prefix(id, bp):
    def nud(self):
        self.first = _parse(bp)
        self.second = None
        return self
    symbol(id).nud = nud

def infix_r(id, bp):
    def led(self, left):
        self.first = left
        self.second = _parse(bp-1)
        return self
    symbol(id, bp).led = led

def constant(id):
    def nud(self):
        self.id = "(const)"
        self.value = id
        return self
    symbol(id).nud = nud

def advance(id=None):
    global token
    if id and token.id != id:
        raise SyntaxError("Expected %r" % id)
    token = next()

symbol("(number)").nud = lambda self: self
symbol("(name)").nud = lambda self: self
symbol("(end)")

infix("=", 10)

def led(self, left):
    self.first = left
    self.second = _parse(0)
    advance("else")
    self.third = _parse(0)
    return self
symbol("if", 20) # ternary
symbol("else")

infix_r("or", 30)
infix_r("and", 40)
prefix("not", 50)

def led(self, left):
    if token.id == "not":
        advance()
        self.id = "is not"
    self.first = left
    self.second = _parse(60)
    return self
symbol("is", 60).led = led
infix("not", 60)

infix("<", 60)
infix("<=", 60)
infix(">", 60)
infix(">=", 60)
infix("<>", 60)
infix("!=", 60)
infix("==", 60)

infix("|", 70)
infix("^", 80)
infix("&", 90)

infix("<<", 100)
infix(">>", 100)

infix("+", 110)
infix("-", 110)
infix("*", 120)
infix("/", 120)
infix("//", 120)
infix("%", 120)

prefix("+", 130)
prefix("-", 130)
prefix("~", 130)

infix_r("**", 140)

def led(self, left):
    if token.id != "(name)":
        SyntaxError("Expected an attribute name.")
    self.first = left
    self.second = token
    advance()
    return self
symbol(".", 150).led = led

def led(self, left):
    self.first = left
    self.second = []
    while True:
        if token.id == "]":
            break
        self.second.append(_parse(0))
        if token.id != ",":
            break
        advance(",")
    advance("]")
    return self
symbol("[", 150).led = led
symbol("]")

def nud(self):
    self.first = _parse(0)
    self.second = None
    advance(")")
    return self
symbol("(", 150).nud = nud
def led(self, left):
    self.first = left
    self.second = []
    if token.id != ")":
        while True:
            self.second.append(_parse(0))
            if token.id != ",":
                break
            advance(",")
    advance(")")
    return self
symbol("(").led = led
symbol(")")

symbol(",")

constant("True")
constant("False")
