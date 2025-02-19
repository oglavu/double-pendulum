
all:
	python src/replace_greek.py src/main.cu build/main.cu
	nvcc build/main.cu -o bin/main.exe -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.37.32822\bin\Hostx64\x64"