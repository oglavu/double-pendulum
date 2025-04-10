#ifndef GL_H
#define GL_H

#include <string>

class Shader {
private:
    unsigned m_programId;
    enum ShaderType { VERTEX = 1, FRAGMENT = 2};

    static std::string readFile(std::string filePath);
    static unsigned compile(const char* str, ShaderType type);
    static unsigned link(const unsigned vertexId, const unsigned fragmentId);

public:
    Shader(std::string vsFileName, std::string fsFileName);

    void use() const;

    void uniform(const std::string, const int) const;
    
};

class StripBufferGL {
private:
    const int lines_per_frame, frames_per_buffer;

    int m_glIndex = -1;

    unsigned m_vbo[2];
    unsigned m_vao;

    mutable size_t m_offset = 0;
    void* m_cudaResource[2] = {0, 0};
    void* m_dInitArray = 0;

    void* cuda_map(int ix);
    void  cuda_fill(int ix);
    void  cuda_unmap(int ix);

public:
    StripBufferGL(int lines_per_frame, void* h_ptr);

    void init();
    
    void update();

    void draw() const;

    ~StripBufferGL();
};


#endif // GL_H