from dtypes import dtype_to_ctype, _fill_dtype_registry

_fill_dtype_registry(respect_windows=False)

class Argument(object):
    def __init__(self, dtype, name):
        self.dtype = dtype
        self.name = name

class ArrayArg(Argument):
    def decltype(self):
        return "GLOBAL_MEM %s *" % (dtype_to_ctype(self.dtype))

    def isarray(self):
        return True

class ScalarArg(Argument):
    def decltype(self):
        return dtype_to_ctype(self.dtype)
    
    def isarray(self):
        return False
