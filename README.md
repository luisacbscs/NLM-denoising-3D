# NLM-denoising-3D
Non-local means denoising filter for 3D images.

- C++ implementation of the non-local means algorithm for image denoising, as described by Buades, Coll and Morel (2005). Implemented for the denoising of NIfTI (.nii) medical images (more specifically, whole-body PET/CT).
- CMakeLists.txt contains the project's configuration. ITK library is necessary as it is used to read and write the images. The build directory of the ITK library must be provided when configuring the project with CMake.
- nlm.cxx contains the implementation of the non-local means filter. Iteration over the image is parallelized for runtime efficiency.
- NLMDenoising.exe is the release version of the implemented solution. The arguments are: path to the image file to denoise, path to the output file (optional), noise standard deviation estimate, decay factor h, kernel size and neighbourhood radius. If none are passed, the user is asked to provide them. 
