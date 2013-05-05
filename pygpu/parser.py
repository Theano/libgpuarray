import re
import tokenize
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

class Compiler(object):
    def __init__(self):
        self.symbol_table = {}
        self.setup_syntax()
        self.reset()

    def reset(self):
        self.token = None
        self.next = None

    def parse(self, program):
        self.next = self._tokenize(program).next
        self.token = self.next()
        statements = []
        while True:
            statements.append(self._parse(0))
            if self.token.id != ",":
                break;
            self.advance(",")
        if self.token.id != "(end)":
            raise SyntaxError("Trailing tokens")
        return statements

    def check(self, statements):
        for tree in statement:
            if (tree.id != '='):
                warnings.warn("top level statement is not an assignment", SyntaxWarning)
            self._check(tree)

    def transform(self, statements):
        return [_transform(tree) for tree in statments]

    def generate(self, statements, symbol_table):
        return ','.join(self._generate(tree) for tree in statements)

    def _check(self, node):
        # assignements are to a variable only
        # (also mark that variable as an output)
        if node.id == '=':
            if node.first.id != '(name)':
                raise SyntaxError('assignment to a non-variable')
            else:
                vname = node.first.value
                if vname not in self.variables:
                    raise SyntaxError("Unknown variable '%s'" % (vname,))
                self.variables[vname].out = True

        # attribute lookup is unsupported
        if node.id == '.':
            raise SyntaxError('attribute lookup is not supported outside of the special indexing syntax')
        # keyword arguments are not supported
        if node.id == '(' and node.second is not None:
            for n in node.second:
                if node.id == '=':
                    raise SyntaxError("keyword arguments are not supported")
        # treat the 'a.i[1]' case here
        if node.id == '[' and node.first.id == '.':
            if node.first.second.id == '(name)' and \
                    node.first.second.value == 'i':
                node.id = 'i['
                node.first = node.first.first
        # add other rules here.


        # recursively check our sub-nodes
        if node.first is not None:
            node.first = self._check(node.first)
        if node.second is not None:
            node.second = self._check(node.second)
        if node.third is not None:
            node.third = self._check(node.third)
        return node

    def _parse(self, rbp):
        t = self.token
        self.token = self.next()
        left = t.nud(self)
        while rbp < self.token.lbp:
            t = self.token
            self.token = self.next()
            left = t.led(self, left)
        return left

    def _tokenize(self, program):
        for t in tokenize.generate_tokens(StringIO(program).next):
            s = None
            if t[0] == tokenize.NUMBER:
                s = self.symbol_table['(number)']()
                s.value = t[1]
            elif t[0] == tokenize.NAME:
                sym = self.symbol_table.get(t[1])
                if sym:
                    s = sym()
                else:
                    s = self.symbol_table['(name)']()
                    s.value = t[1]
            elif t[0] == tokenize.ENDMARKER:
                s = self.symbol_table['(end)']()
            elif t[0] == tokenize.OP:
                try:
                    s = self.symbol_table[t[1]]()
                except KeyError:
                    raise SyntaxError('Unknown operator: ' + t[1])
            yield s

    class symbol_base(object):
        id = None
        value = None
        first = second = third = None

        def nud(self, parser):
            raise SyntaxError('nud')

        def led(self, parser, left):
            raise SyntaxError('led')

        def __repr__(self):
            if self.id == "(name)" or self.id == "(number)":
                return "(%s %s)" % (self.id[1:-1], self.value)
            else:
                return "(" + " ".join(map(repr, filter(None, [self.id, \
                                 self.first, self.second, self.third]))) + ")"

    def symbol(self, id, bp=0):
        try:
            s = self.symbol_table[id]
        except KeyError:
            class s(self.symbol_base):
                pass
            s.__name__ = "symbol-" + id
            s.id = id
            s.lbp = bp
            self.symbol_table[id] = s
        else:
            s.lbp = max(bp, s.lbp)
        return s

    def infix(self, id, bp):
        def led(self, parser, left):
            self.first = left
            self.second = parser._parse(bp)
            return self
        self.symbol(id, bp).led = led

    def prefix(self, id, bp):
        def nud(self, parser):
            self.first = parser._parse(bp)
            self.second = None
            return self
        self.symbol(id).nud = nud

    def infix_r(self, id, bp):
        def led(self, parser, left):
            self.first = left
            self.second = parser._parse(bp-1)
            return self
        self.symbol(id, bp).led = led

    def constant(self, id):
        def nud(self):
            self.id = "(const)"
            self.value = id
            return self
        self.symbol(id).nud = nud

    def advance(self, id=None):
        if id and self.token.id != id:
            raise SyntaxError("Expected %r" % id)
        self.token = self.next()

    def setup_syntax(self):
        self.symbol("(number)").nud = lambda self, parser: self
        self.symbol("(name)").nud = lambda self, parser: self
        self.symbol("(end)")

        self.infix("=", 10)

        def led(self, parser, left):
            self.first = left
            self.second = parser._parse(0)
            parser.advance("else")
            self.third = parser._parse(0)
            return self
        self.symbol("if", 20).led = led # ternary
        self.symbol("else")

        self.infix_r("or", 30)
        self.infix_r("and", 40)
        self.prefix("not", 50)

        def led(self, parser, left):
            if parser.token.id == "not":
                parser.advance()
                self.id = "is not"
            self.first = left
            self.second = parser._parse(60)
            return self
        self.symbol("is", 60).led = led

        self.infix("<", 60)
        self.infix("<=", 60)
        self.infix(">", 60)
        self.infix(">=", 60)
        self.infix("<>", 60)
        self.infix("!=", 60)
        self.infix("==", 60)

        self.infix("|", 70)
        self.infix("^", 80)
        self.infix("&", 90)

        self.infix("<<", 100)
        self.infix(">>", 100)

        self.infix("+", 110)
        self.infix("-", 110)
        self.infix("*", 120)
        self.infix("/", 120)
        self.infix("//", 120)
        self.infix("%", 120)

        self.prefix("+", 130)
        self.prefix("-", 130)
        self.prefix("~", 130)

        self.infix_r("**", 140)

        def led(self, parser, left):
            if parser.token.id != "(name)":
                raise SyntaxError("Expected an attribute name.")
            self.first = left
            self.second = parser.token
            parser.advance()
            return self
        self.symbol(".", 150).led = led

        def led(self, parser, left):
            self.first = left
            self.second = []
            while True:
                if parser.token.id == "]":
                    break
                self.second.append(parser._parse(0))
                if parser.token.id != ",":
                    break
                parser.advance(",")
            parser.advance("]")
            return self
        self.symbol("[", 150).led = led
        self.symbol("]")

        def nud(self, parser):
            self.first = parser._parse(0)
            self.second = None
            parser.advance(")")
            return self
        def led(self, parser, left):
            self.first = left
            self.second = []
            if parser.token.id != ")":
                while True:
                    self.second.append(parser._parse(0))
                    if parser.token.id != ",":
                        break
                    parser.advance(",")
            parser.advance(")")
            return self
        self.symbol("(", 150).nud = nud
        self.symbol("(").led = led
        self.symbol(")")

        self.symbol(",")

        self.constant("True")
        self.constant("False")
