cmake_minimum_required(VERSION 3.27)

project(NLMProject)
  
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
  
add_executable(NLMProject nlm.cxx)
  
target_link_libraries(NLMProject ${ITK_LIBRARIES})
