cmake_minimum_required(VERSION 3.20)

project(vkgsplat)

find_package(Vulkan REQUIRED)

set(CMAKE_CXX_STANDARD 23)
set(CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(ROSE_ENABLE_TESTING OFF CACHE BOOL "")
set(ROSE_BUILD_APPS     OFF CACHE BOOL "")
add_subdirectory(extern/Rose)

file(TO_CMAKE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src" SHADER_INCLUDE_DIR)
set_source_files_properties(
    "${CMAKE_CURRENT_SOURCE_DIR}/extern/Rose/src/Rose/Core/ShaderModule.cpp" PROPERTIES COMPILE_FLAGS
    "-DDEFAULT_SHADER_INCLUDE_PATHS=\\\"${SHADER_INCLUDE_DIR}\\\"," )

add_executable(vkgsplat
    src/App.cpp
)
set_target_properties(vkgsplat PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(vkgsplat PUBLIC Vulkan::Vulkan)
target_link_libraries(vkgsplat PUBLIC glm)
target_link_libraries(vkgsplat PRIVATE RoseLib)

target_compile_definitions(vkgsplat PUBLIC WIN32_LEAN_AND_MEAN _USE_MATH_DEFINES GLM_FORCE_XYZW_ONLY IMGUI_DEFINE_MATH_OPERATORS VULKAN_HPP_NO_STRUCT_CONSTRUCTORS)
