include(cc_library)

cc_library(
  NAME
    llumnix
  HDRS
    llumnix_instance_mgr.h
    llumnix_load.h
  SRCS
    llumnix_instance_mgr.cpp
  DEPS
    :common
    :managers
    :request
    glog::glog
    nlohmann_json::nlohmann_json
)
