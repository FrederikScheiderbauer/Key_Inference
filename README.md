Path tracing project build on top of "https://github.com/KhronosGroup/glTF-Sample-Viewer" to experiment with Shader Execution Reordering Technology of modern NVidia GPUs. Performs at runtime Thread Reordering to optimize path tracing performance by inferring optimal sorting keys from the ray data and scene context.

To Clone he repository with the necessary submodules run:

git clone --recurse-submodules https://github.com/Motoerdeath/Key_Inference_For_SER.git 

Compilation:

-cd Key_Inference_For_SER/Prototype 
-mkdir build 
-cd build 
-cmake .. 
-cmake --build . 

the executable can then be found in Key_Inference_For_SER/bin_x64/Debug/

