#define _CRT_SECURE_NO_WARNINGS
#include <sys/stat.h>
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>

#ifdef _MSC_VER
#include <io.h>
#define open _open
#define mktemp _mktemp
#else
#define O_BINARY 0
#endif

int mkstemp(char *path) {
    char *tmp;
    int res;
    int tries = 3;

    do {
        tmp = mktemp(path);
        if (tmp == NULL) return -1;
        res = open(path, O_CREAT|O_EXCL|O_RDWR|O_BINARY, S_IREAD|S_IWRITE);
        if (res != -1 || errno != EEXIST)
            return res;
    } while (--tries);

    errno = EEXIST;
    return -1;
}
