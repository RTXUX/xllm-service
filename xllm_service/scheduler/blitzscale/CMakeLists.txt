include(cc_library)

cc_library(
  NAME
    blitzscale
  HDRS
    blitzscale_instance_mgr.h
  SRCS
    blitzscale_instance_mgr.cpp
  DEPS
    :common
    :managers
    :request
    glog::glog
    nlohmann_json::nlohmann_json
)
