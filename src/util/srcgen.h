/* Include Guards */
#ifndef SRCGEN_H
#define SRCGEN_H


/* Includes */
#include "util/strb.h"


/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif
#ifdef CONFUSE_EMACS
}
#endif



/* Data Structure Prototypes & Typedefs */
struct srcb;
typedef struct srcb srcb;



/* Enumerations */
enum srcb_state{
	SRCB_STATE_NONE,
	SRCB_STATE_INLIST,
};
typedef enum srcb_state srcb_state;



/* Data Structures */

/**
 * @brief The srcb struct
 * 
 * The Source Code Buffer. Augments strb with C-like language generation tools.
 */

struct srcb{
	strb*       s;
	srcb_state  state;
	int         numElems;
	const char* sep;
	const char* empty;
};



/* Functions */
static inline  void srcbInit  (srcb* s, strb* sb){
	s->s          = sb;
	s->state      = SRCB_STATE_NONE;
	s->numElems   = 0;
}
static inline  void srcbBeginList(srcb* s, const char* sep, const char* empty){
	s->state      = SRCB_STATE_INLIST;
	s->numElems   = 0;
	s->sep        = sep;
	s->empty      = empty;
}
static inline  void srcbEndList(srcb* s){
	if(s->numElems == 0){
		strb_appends(s->s, s->empty);
	}
	
	s->state      = SRCB_STATE_NONE;
	s->numElems   = 0;
	s->sep        = "";
	s->empty      = "";
}
static inline  void srcbAppendElemv(srcb* s, const char *f, va_list ap){
	if(s->numElems > 0){
		strb_appends(s->s, s->sep);
	}
	
	strb_appendv(s->s, f, ap);
	
	s->numElems++;
}
static inline  void srcbAppendElemf(srcb* s, const char *f, ...){
	va_list ap;
	va_start(ap, f);
	srcbAppendElemv(s, f, ap);
	va_end(ap);
}
static inline  void srcbAppends(srcb* s, const char *f){
	strb_appends(s->s, f);
}
static inline  void srcbAppendf(srcb* s, const char *f, ...){
	va_list ap;
	va_start(ap, f);
	strb_appendv(s->s, f, ap);
	va_end(ap);
}


/* End Extern "C" Guard */
#ifdef __cplusplus
}
#endif

#endif
