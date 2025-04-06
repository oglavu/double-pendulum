
#ifndef MMF_H
#define MMF_H

#if _WIN32
#include <windows.h>
#elif __linux__
#include <cstddef>
#include <cstdint>
#endif


namespace mmf {

    struct mmap_t {
#if _WIN32
        HANDLE file;
        HANDLE hMap;
#elif __linux__
        int file;
#endif
        size_t sz;
        void*  h_array;
    };

    int map_file_rd(const char* filename, struct mmf::mmap_t* map);

    int map_file_wr(const char* filename, struct mmf::mmap_t* maps, const uint64_t n, const size_t seg_size);

    void unmap_file(struct mmf::mmap_t* maps, uint64_t n);
}

#endif // MMF_H