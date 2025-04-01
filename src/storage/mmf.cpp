#include "mmf.hpp"

int mmf::map_file_rd(const char* filename, struct mmf::mmap_t* map) {

    map->file = CreateFileA(
        filename, GENERIC_READ | GENERIC_WRITE, 0, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0
    );
    if (map->file == INVALID_HANDLE_VALUE) {
        std::cerr << "Error: Opening read file: " << GetLastError() << std::endl;
        return -1;
    }

    SetFilePointer(map->file, map->sz, 0, FILE_BEGIN);
    SetEndOfFile(map->file);

    // create mapping
    map->hMap = CreateFileMapping(
        map->file, 0, PAGE_READWRITE, 0, map->sz, 0
    );
    if (!map->hMap) {
        std::cerr << "Error: Creating read file mapping: " << GetLastError() << std::endl;
        CloseHandle(map->file);
        return -2;
    }

    map->h_array = MapViewOfFile(map->hMap, FILE_MAP_WRITE, 0, 0, map->sz);
    if (!map->h_array) {
        std::cerr << "Error: mapping view of read file: " << GetLastError() << std::endl;
        CloseHandle(map->hMap);
        CloseHandle(map->file);
        return -3;
    }

    return 0;
}

int mmf::map_file_wr(
    const char* filename, 
    struct mmf::mmap_t* maps, 
    const uint64_t n, 
    const size_t seg_size
) {
    HANDLE file = CreateFileA(
        filename, GENERIC_READ | GENERIC_WRITE, 0, 0, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, 0
    );
    if (file == INVALID_HANDLE_VALUE) {
        std::cerr << "Error: Opening write file: " << GetLastError() << std::endl;
        return -1;
    }

    const uint64_t sz  = n * seg_size;
    const uint32_t szL = (uint32_t)sz;
    const uint32_t szH = (uint32_t)(sz >> 32);

    SetFilePointer(file, szL, (long*)&szH, FILE_BEGIN);
    SetEndOfFile(file);

    // create mapping
    HANDLE hMap = CreateFileMapping(
        file, 0, PAGE_READWRITE, szH, szL, 0
    );
    if (!hMap) {
        std::cerr << "Error: Creating write file mapping: " << GetLastError() << std::endl;
        CloseHandle(file);
        return -2;
    }

    uint64_t offset = 0;
    for (uint64_t i=0; offset < sz; ++i) {
        maps[i].file = file;
        maps[i].hMap = hMap;
        maps[i].sz = seg_size;
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
        offset += seg_size; 
    }

    return 0;
}

void mmf::unmap_file(struct mmf::mmap_t* maps, uint64_t n) {
    for (uint64_t i=0; i<n; i++) {
        FlushViewOfFile(maps[i].h_array, maps[i].sz);
        UnmapViewOfFile(maps[i].h_array);
    }
    CloseHandle(maps[0].hMap);
    CloseHandle(maps[0].file);
}
