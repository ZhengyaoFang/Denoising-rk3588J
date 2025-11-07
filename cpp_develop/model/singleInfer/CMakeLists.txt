cmake_minimum_required(VERSION 3.5.0)
project(SingleFrameInfer)

# 查找 HailoRT SDK（保持原有逻辑）
find_package(HailoRT 4.18.0 EXACT REQUIRED)

# 查找 OpenCV（新增）
find_package(OpenCV REQUIRED)

# 创建可执行文件
add_executable(single_infer single_infer.cpp)

# 链接 HailoRT 与 OpenCV
target_link_libraries(single_infer
    PRIVATE
        HailoRT::libhailort
        ${OpenCV_LIBS}
)

# Windows 兼容（保持原逻辑）
if(WIN32)
    target_compile_options(single_infer PRIVATE
        /DWIN32_LEAN_AND_MEAN
        /DNOMINMAX
        /wd4201 /wd4251
    )
endif()

# 设置 C++ 标准（建议提升到 17，以支持现代写法）
set_target_properties(single_infer PROPERTIES CXX_STANDARD 17)
