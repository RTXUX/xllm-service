#pragma once
#include "brpc/coroutine.h"

namespace xllm_service {

using CoroTask  = brpc::experimental::Awaitable<void>;
using AwaitDone = brpc::experimental::AwaitableDone;
using Coroutine = brpc::experimental::Coroutine;

}  // namespace xllm_service
