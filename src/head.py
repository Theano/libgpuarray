# Used to generate the string tables to embed the cluda headers.
# Usage: python head.py <file>
# This will output <file>.c

def wrt(f, n, b):
    f.write(b',')
    n += 1
    if n > 10:
        f.write(b'\n')
        n = 0
    else:
        f.write(b' ')
    f.write(b"0x%02x" % (b,))
    return n


def convert(src, dst):
    src_name = src.replace('.', '_')
    with open(src, 'rb') as f:
        src_data = f.read()
    with open(dst, 'wb') as f:
        f.write(b'static const char %s[] = {\n' % (src_name.encode('utf-8'),))
        first = True
        n = 0
        for b in bytearray(src_data):
            if b == 0:
                raise ValueError('NUL in file')
            if first:
                f.write(b"0x%02x" % (b,))
                first = False
            else:
                n = wrt(f, n, b)
        wrt(f, n, 0)
        f.write(b'};\n')

if __name__ == '__main__':
    import sys
    convert(sys.argv[1], sys.argv[1] + '.c')
