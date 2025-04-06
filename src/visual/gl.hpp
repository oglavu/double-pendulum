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

    unsigned m_vbo;
    unsigned m_vao;

    void* m_cudaResource = 0;
    void* m_dcudaArray = 0;

public:
    StripBufferGL(int lines_per_frame);
    
    void* cuda_map();

    void draw() const;

    void cuda_unmap();

    ~StripBufferGL();
};


#endif // GL_H