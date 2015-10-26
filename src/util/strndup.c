/* strndup.c
*
*/

#include <stdlib.h>
#include <string.h>

// strndup not defined for Microsoft Visual Studio
#ifdef _MSC_VER
char *strndup(const char *s, size_t size)
{
  char *res = NULL;
  if (s != NULL)
  {
    size = strnlen(s, size);
    if (size > 0)
    {
      res = malloc(size + 1);
      if (res != NULL)
      {
        memcpy(res, s, size);
        res[size] = '\0';
      }
      // else malloc returns NULL and sets errno = ENOMEM
    }
  }
  return res;
}
#endif