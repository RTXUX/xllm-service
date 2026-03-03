include(cc_library)

cc_library(
  NAME
    resource_model
  HDRS
    resource_model.h
    linear_resource_model.h
  SRCS
    linear_resource_model.cpp
  DEPS
    :common
)
