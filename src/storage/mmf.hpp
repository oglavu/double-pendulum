
#ifndef MMF_H
#define MMF_H

#include <windows.h>
#include <iostream>

namespace mmf {

    struct mmap_t {
        HANDLE file;
        HANDLE hMap;
        size_t sz;
        void*  h_array;
    };

    int map_file_rd(const char* filename, struct mmf::mmap_t* map);

    int map_file_wr(const char* filename, struct mmf::mmap_t* maps, const uint64_t n, const size_t seg_size);

    void unmap_file(struct mmf::mmap_t* maps, uint64_t n);
}

#endif // MMF_H