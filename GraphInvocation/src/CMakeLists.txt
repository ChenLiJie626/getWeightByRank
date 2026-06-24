cmake_minimum_required(VERSION 3.5.1)

project(ge_execute_get_weight_by_rank)

add_compile_options(-std=c++11)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../output")

set(INC_PATH $ENV{DDK_PATH})
if (NOT DEFINED ENV{DDK_PATH})
    set(INC_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif()

set(LIB_PATH $ENV{NPU_HOST_LIB})
if (NOT DEFINED ENV{NPU_HOST_LIB})
    string(TOLOWER "${CMAKE_SYSTEM_NAME}" SYSTEM_NAME_LOWER)
    set(LIB_PATH "/usr/local/Ascend/ascend-toolkit/latest/${CMAKE_SYSTEM_PROCESSOR}-${SYSTEM_NAME_LOWER}/lib64")
endif()

set(CUST_PROTO_PATH "${INC_PATH}/opp/vendors/customize/op_proto")
set(CUST_TILING_PATH "${INC_PATH}/opp/vendors/customize/op_impl/ai_core/tbe/op_tiling")
set(DRIVER_LIB_PATH "/usr/local/Ascend/driver/lib64/driver")
set(COMPILER_LIB_PATH "${INC_PATH}/compiler/lib64")

include_directories(
    ${INC_PATH}/include
    ${INC_PATH}/opp/built-in/op_proto/inc
    ${CUST_PROTO_PATH}/inc
)

link_directories(
    ${LIB_PATH}
    ${COMPILER_LIB_PATH}
    ${CUST_PROTO_PATH}/lib/linux/aarch64
    ${CUST_TILING_PATH}/lib/linux/aarch64
    ${DRIVER_LIB_PATH}
)

add_executable(execute_get_weight_by_rank_graph
    main.cpp
)

target_link_libraries(execute_get_weight_by_rank_graph
    ge_runner
    graph
    graph_base
    ge_common
    ge_common_base
    ascendcl
    register
    cust_opsproto_rt2.0
    cust_opmaster_rt2.0
    stdc++
)

target_link_options(execute_get_weight_by_rank_graph PRIVATE
    -Wl,-rpath-link,${DRIVER_LIB_PATH}
)
