#include <sys/types.h>

typedef struct _compyte_type {
  const char *name;
  size_t size;
} compyte_type;

enum compyte_typecode {
  GA_CHAR = 0,
  GA_UCHAR = 1,
  GA_SHORT = 2,
  GA_USHORT = 3,
  GA_INT = 4,
  GA_UINT = 5,
  GA_LONG = 6,
  GA_ULONG = 7,
  GA_FLOAT = 8,
  GA_DOUBLE = 9,
  GA_CHAR2 = 10,
  GA_UCHAR2 = 11,
  GA_CHAR3 = 12,
  GA_UCHAR3 = 13,
  GA_CHAR4 = 14,
  GA_UCHAR4 = 15,
  GA_SHORT2 = 16,
  GA_USHORT2 = 17,
  GA_SHORT3 = 18,
  GA_USHORT3 = 19,
  GA_SHORT4 = 20,
  GA_USHORT4 = 21,
};

extern compyte_type builtin_types[];
