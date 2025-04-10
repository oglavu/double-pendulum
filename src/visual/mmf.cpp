#include "mmf.hpp"
#include <iostream>

#if __linux__
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

int mmf::map_file_rd(const char* filename, struct mmf::mmap_t* map) {

    // open file
#if _WIN32
    map->file = CreateFileA(
        filename, GENERIC_READ | GENERIC_WRITE, 0, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0
    );
    if (map->file == INVALID_HANDLE_VALUE) {
        std::cerr << "Error: Opening read file: " << GetLastError() << std::endl;
        return -1;
    }
#elif __linux__
    map->file = open(filename, O_RDWR | O_CREAT, 0777);
    if (map->file < 0) {
        std::cerr << "Error: Opening read file: " << errno << std::endl;
        return -1;
    }
#endif

    // resize file
#if _WIN32
    SetFilePointer(map->file, map->sz, 0, FILE_BEGIN);
    SetEndOfFile(map->file);
#elif __linux__
    if (ftruncate(map->file, map->sz) == -1) {
        std::cerr << "Error: Resizing file: " << errno << std::endl;
        close(map->file);
        return -2;
    }
#endif

    // create mapping
#if _WIN32
    map->hMap = CreateFileMapping(
        map->file, 0, PAGE_READWRITE, 0, map->sz, 0
    );
    if (!map->hMap) {
        std::cerr << "Error: Creating read file mapping: " << GetLastError() << std::endl;
        CloseHandle(map->file);
        return -3;
    }

    map->h_array = MapViewOfFile(map->hMap, FILE_MAP_WRITE, 0, 0, map->sz);
    if (!map->h_array) {
        std::cerr << "Error: mapping view of read file: " << GetLastError() << std::endl;
        CloseHandle(map->hMap);
        CloseHandle(map->file);
        return -4;
    }
#elif __linux__
    map->h_array = mmap(0, map->sz, PROT_READ | PROT_WRITE, MAP_SHARED, map->file, 0);
    if (map->h_array == MAP_FAILED) {
        std::cerr << "Error: Mapping file: " << errno << std::endl;
        close(map->file);
        return -3;
    }
#endif

    printf("MMF: File %s read successfully.\n", filename); 

    return 0;
}

int mmf::map_file_wr(
    const char* filename, 
    struct mmf::mmap_t* maps, 
    const uint64_t n, 
    const size_t seg_size
) {
    // open file
#if _WIN32
    HANDLE file = CreateFileA(
        filename, GENERIC_READ | GENERIC_WRITE, 0, 0, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, 0
    );
    if (file == INVALID_HANDLE_VALUE) {
        std::cerr << "Error: Opening write file: " << GetLastError() << std::endl;
        return -1;
    }
#elif __linux__
    int file = open(filename, O_RDWR | O_CREAT, 0777); // Open file for reading and writing
    if (file < 0) {
        std::cerr << "Error: Opening read file: " << errno << std::endl;
        return -1;
    }
#endif

    const uint64_t sz  = n * seg_size;
    const uint32_t szL = (uint32_t)sz;
    const uint32_t szH = (uint32_t)(sz >> 32);

    // resize file
#if _WIN32
    SetFilePointer(file, szL, (long*)&szH, FILE_BEGIN);
    SetEndOfFile(file);
#elif __linux__
    if (ftruncate(file, n * seg_size) == -1) {
        std::cerr << "Error: Resizing file: " << errno << std::endl;
        close(file);
        return -3;
    }
#endif

    // create mapping
#if _WIN32
    HANDLE hMap = CreateFileMapping(
        file, 0, PAGE_READWRITE, szH, szL, 0
    );
    if (!hMap) {
        std::cerr << "Error: Creating write file mapping: " << GetLastError() << std::endl;
        CloseHandle(file);
        return -2;
    }
#endif

    uint64_t offset = 0;
    for (uint64_t i=0; offset < sz; ++i) {
        maps[i].file = file;
        maps[i].sz = seg_size;
        
#if _WIN32
        maps[i].hMap = hMap;
        maps[i].h_array = MapViewOfFile(hMap, FILE_MAP_WRITE, (offset >> 32), (uint32_t)offset, seg_size);
        
        if (!maps[i].h_array) {
            std::cerr << "Error: mapping view of write file on "<< i << ". segment: " << GetLastError() << std::endl;
            for (; i>0; --i) {
                UnmapViewOfFile(maps[i].h_array); 
            }
            CloseHandle(hMap);
            CloseHandle(file);
            return -3;
        }
#elif __linux__
        maps[i].h_array = mmap(nullptr, seg_size, PROT_READ | PROT_WRITE, MAP_SHARED, file, offset);
                
        if (maps[i].h_array == MAP_FAILED) {
            std::cerr << "Error: Mapping view of write file on " << i << ". segment: " << errno << std::endl;
            for (; i > 0; --i) {
                munmap(maps[i - 1].h_array, seg_size);
            }
            close(file);
            return -3;
        }
#endif
        offset += seg_size; 
    }

    return 0;
}

void mmf::unmap_file(struct mmf::mmap_t* maps, uint64_t n) {
    for (uint64_t i=0; i<n; i++) {
#if _WIN32
        FlushViewOfFile(maps[i].h_array, maps[i].sz);
        UnmapViewOfFile(maps[i].h_array);
#elif __linux__
        msync(maps[i].h_array, maps[i].sz, MS_SYNC);
        munmap(maps[i].h_array, maps[i].sz);
#endif    
    }
    
#if _WIN32
    CloseHandle(maps[0].hMap);
    CloseHandle(maps[0].file);
#elif __linux__
    close(maps[0].file);
#endif
}
