include(cc_library)

set(CMAKE_CXX_STANDARD 17)      # cpprestsdk is not C++20-clean
add_subdirectory(cpprestsdk)
set(CMAKE_CXX_STANDARD 20)      # restore for all other subdirs
set(CPPREST_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpprestsdk/Release/include)
set(CPPREST_LIB ${CMAKE_BINARY_DIR}/third_party/cpprestsdk/Release/Binaries/libcpprest.a)
add_subdirectory(etcd_cpp_apiv3)
add_subdirectory(brpc)
add_subdirectory(minja)
add_subdirectory(sentencepiece)
add_subdirectory(smhasher/src)