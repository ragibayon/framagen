
char *cstrcat(char *dest, const char *src) {
    char *p = dest;

    while (*p != '\0') {
        p++;
    }

    while (*src != '\0') {
        *p = *src;
        p++;
        src++;
    }

    *p = '\0';
    return dest;
}

int main(void) {
    char *dest;
    char *src;

    cstrcat(dest, src);
    return 0;
}
