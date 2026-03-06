include(cc_library)

cc_library(
  NAME
    serverless_llm
  HDRS
    serverless_llm_instance_mgr.h
    serverless_llm_store_mgr.h
  SRCS
    serverless_llm_instance_mgr.cpp
    serverless_llm_store_mgr.cpp
  DEPS
    :common
    :managers
    :request
    glog::glog
    nlohmann_json::nlohmann_json
)
