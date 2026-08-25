cmake_minimum_required(VERSION 3.14)
project(OnnxInference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find libcurl (for HTTP requests to LLM API)
find_package(CURL REQUIRED)

find_package(nlohmann_json REQUIRED)

# ONNX Runtime path
set(ONNXRUNTIME_ROOT "E:/聯成電腦/onnxruntime-win-x64-1.27.0" CACHE PATH "Path to ONNX Runtime")

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_IMPLIB "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib"
    IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include"
)

add_executable(inference main.cpp)

target_link_libraries(inference
    PRIVATE
    ${OpenCV_LIBS}
    onnxruntime
    CURL::libcurl
    nlohmann_json::nlohmann_json
)

# Copy ONNX DLLs
add_custom_command(TARGET inference POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
        "$<TARGET_FILE_DIR:inference>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll"
        "$<TARGET_FILE_DIR:inference>"
)