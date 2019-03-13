#ifndef REACTIVETRANSPORTGPU_STRING_FORMAT_H
#define REACTIVETRANSPORTGPU_STRING_FORMAT_H

std::string string_format(const std::string fmt, ...) {
    int n, size=100;
    std::string str;
    va_list ap;

    while (1) {
        str.resize(size);
        va_start(ap, fmt);
        int n = vsnprintf(&str[0], size, fmt.c_str(), ap);
        va_end(ap);

        if (n > -1 && n < size)
            return str;
        if (n > -1)
            size = n + 1;
        else
            size *= 2;
    }
}

#endif //REACTIVETRANSPORTGPU_STRING_FORMAT_H
