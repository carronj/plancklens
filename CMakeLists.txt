cmake_minimum_required(VERSION 3.17.2...3.29)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C Fortran)

find_package(Python COMPONENTS Development.Module NumPy REQUIRED)

# F2PY include path
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "import numpy.f2py; print(numpy.f2py.get_include())"
  OUTPUT_VARIABLE F2PY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)

# fortranobject object library
add_library(fortranobject OBJECT "${F2PY_INCLUDE_DIR}/fortranobject.c")
target_link_libraries(fortranobject PUBLIC Python::NumPy)
target_include_directories(fortranobject PUBLIC "${F2PY_INCLUDE_DIR}")
set_property(TARGET fortranobject PROPERTY POSITION_INDEPENDENT_CODE ON)

# --- plancklens.n1.n1f ---
# Auto-generate wrappers
add_custom_command(
  OUTPUT n1fmodule.c n1f-f2pywrappers.f
  DEPENDS plancklens/n1/n1f.f90
  COMMAND "${Python_EXECUTABLE}" -m numpy.f2py
          "${CMAKE_CURRENT_SOURCE_DIR}/plancklens/n1/n1f.f90"
          -m n1f --lower
  VERBATIM)

# Compile extension in the correct namespace
python_add_library(
  n1f MODULE
  "${CMAKE_CURRENT_BINARY_DIR}/n1fmodule.c"
  "${CMAKE_CURRENT_BINARY_DIR}/n1f-f2pywrappers.f"
  "${CMAKE_CURRENT_SOURCE_DIR}/plancklens/n1/n1f.f90"
  WITH_SOABI)

target_link_libraries(n1f PRIVATE fortranobject)

# Ensure the .so is placed in the source package directory
set_target_properties(n1f PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/plancklens/n1")

# --- plancklens.wigner.wigner ---
# Auto-generate wrappers
add_custom_command(
  OUTPUT wignersmodule.c wigners-f2pywrappers2.f90
  DEPENDS plancklens/wigners/wigners.f90
  COMMAND "${Python_EXECUTABLE}" -m numpy.f2py
          "${CMAKE_CURRENT_SOURCE_DIR}/plancklens/wigners/wigners.f90"
          -m wigners --lower 
  VERBATIM)

# Compile extension in the correct namespace
python_add_library(
  wigners MODULE
  "${CMAKE_CURRENT_BINARY_DIR}/wignersmodule.c"
  "${CMAKE_CURRENT_BINARY_DIR}/wigners-f2pywrappers2.f90"
  "${CMAKE_CURRENT_SOURCE_DIR}/plancklens/wigners/wigners.f90"
  WITH_SOABI)

target_link_libraries(wigners PRIVATE fortranobject)

# Ensure the .so is placed in the source package directory
set_target_properties(wigners PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/plancklens/wigners")

