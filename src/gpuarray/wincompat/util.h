#ifndef WINCOMPAT_UTIL
#define WINCOMPAT_UTIL

#ifdef _MSC_VER
    /* MSVC 2008 does not support "inline". */
    #ifndef inline
        #define inline __inline
    #endif
    #ifndef snprintf
        #define snprintf _snprintf
    #endif
    #ifndef strdup
        #define strdup _strdup
    #endif
    #ifndef alloca
        #define alloca _alloca
    #endif
#endif

#endif
