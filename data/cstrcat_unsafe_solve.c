
#include <string.h>

/*@
  requires valid_read_string(dest);
  requires valid_read_string(src);

  requires \valid(
    dest + (0 .. strlen(dest) + strlen(src))
  );

  requires \separated(
    dest + (0 .. strlen(dest) + strlen(src)),
    src + (0 .. strlen(src))
  );

  requires dest[strlen(dest)] == '\0';
  requires src[strlen(src)] == '\0';

  requires \forall integer i;
    0 <= i < strlen(dest) ==> dest[i] != '\0';

  requires \forall integer i;
    0 <= i < strlen(src) ==> src[i] != '\0';

  assigns dest[
    strlen(dest) .. strlen(dest) + strlen(src)
  ];
  assigns \result \from dest;

  ensures \result == dest;

  ensures dest[
    \old(strlen(dest)) + \old(strlen(src))
  ] == '\0';

  ensures \forall integer i;
    0 <= i < \old(strlen(src)) ==>
    dest[\old(strlen(dest)) + i] == \old(src[i]);
*/
char *cstrcat(char *dest, const char *src) {
    char *p = dest;

    /*@
      loop invariant
        0 <= p - dest <= strlen(dest);

      loop invariant
        \forall integer i;
          0 <= i < strlen(dest) ==> dest[i] != '\0';

      loop invariant dest[strlen(dest)] == '\0';

      loop assigns p;

      loop variant strlen(dest) - (p - dest);
    */
    while (*p != '\0') {
        p++;
    }

    //@ assert p == dest + \at(strlen(dest), Pre);

    //@ ghost char *append_begin = p;
    //@ ghost const char *src_begin = src;

    /*@
      loop invariant
        0 <= src - src_begin <= \at(strlen(src), Pre);

      loop invariant
        p == append_begin + (src - src_begin);

      loop invariant
        \valid(append_begin +
          (0 .. \at(strlen(src), Pre)));

      loop invariant
        \valid_read(src_begin +
          (0 .. \at(strlen(src), Pre)));

      loop invariant
        src_begin[\at(strlen(src), Pre)] == '\0';

      loop invariant
        \forall integer i;
          0 <= i < \at(strlen(src), Pre) ==>
          src_begin[i] != '\0';

      loop invariant
        \forall integer i;
          0 <= i < src - src_begin ==>
          append_begin[i] == src_begin[i];

      loop assigns
        p,
        src,
        append_begin[0 .. \at(strlen(src), Pre) - 1];

      loop variant
        \at(strlen(src), Pre) - (src - src_begin);
    */
    while (*src != '\0') {
        *p++ = *src++;
    }

    *p = '\0';
    return dest;
}

int main(void) {
    char dest[32] = "Hello, ";
    const char src[] = "world!";

    cstrcat(dest, src);
    return 0;
}
