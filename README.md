# opencl-ssa
Gillespie Stochastic Simulation Algorithm (GSSA) implementation using OpenCL

# notes
- Adapted from the hello world example code by Apple
- Used TinyMT RNG v1.1.1 (http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/TINYMT/index.html)
- Implementation is not general for arbitrary biochemical reaction networks. 
- It's a simple network of "fast reversible isomerization process".

# build
>> gcc ssa_opencl.c -o ssa_opencl -I .   -lOpenCL -lm
