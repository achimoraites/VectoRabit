# CMakeList.txt : CMake project for VectoRabit, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)
include(FetchContent)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("VectoRabit")

# Add source to this project's executable.
add_executable (VectoRabit "VectoRabit.cpp" "BallTree.cpp" "VectoRabit.h" "BallTree.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET VectoRabit PROPERTY CXX_STANDARD 20)
endif()

set(CMAKE_Fortran_COMPILER "NOTFOUND")

# Find Crow
find_package(Crow REQUIRED)



# Download Eigen
FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
)
FetchContent_MakeAvailable(Eigen)

target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen Crow::Crow)