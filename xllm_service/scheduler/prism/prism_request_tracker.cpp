#include "scheduler/prism/prism_request_tracker.h"

#include <algorithm>
#include <glog/logging.h>

namespace xllm_service {

double PrismRequestTracker::now_seconds() {
  auto now = std::chrono::system_clock::now();
  return std::chrono::duration<double>(now.time_since_epoch()).count();
}

void PrismRequestTracker::enqueue_req(const std::string& model,
                                       std::shared_ptr<PrismReq> req) {
  std::lock_guard<std::mutex> lock(mutex_);
  req->state = PrismReqState::WAITING;
  if (req->arrival_time <= 0.0) {
    req->arrival_time = now_seconds();
  }
  req->remaining_time_budget = req->slo;
  req->priority = req->arrival_time + req->slo - req->profiled_prefill_time;

  auto& queue = model_queues_[model];
  if (queue.model_name.empty()) {
    queue.model_name = model;
  }
  // Insert sorted by priority (lower = more urgent = earlier in deque)
  auto insert_pos = std::lower_bound(
      queue.waiting_reqs.begin(), queue.waiting_reqs.end(), req,
      [](const std::shared_ptr<PrismReq>& a,
         const std::shared_ptr<PrismReq>& b) {
        return a->priority < b->priority;
      });
  queue.waiting_reqs.insert(insert_pos, req);
  queue.last_active_time = now_seconds();
  all_reqs_[req->rid] = req;

  DLOG(INFO) << "Prism: enqueue request " << req->rid
             << " for model " << model
             << " prompt_len=" << req->prompt_len
             << " priority=" << req->priority;
}

void PrismRequestTracker::start_running(const std::string& rid,
                                         const std::string& instance_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = all_reqs_.find(rid);
  if (it == all_reqs_.end()) {
    LOG(WARNING) << "Prism: start_running called for unknown request " << rid;
    return;
  }
  auto& req = it->second;
  req->state = PrismReqState::RUNNING;
  req->start_running_time = now_seconds();
  req->instance_name = instance_name;

  // Move from waiting to running
  auto& queue = model_queues_[req->model];
  auto wit = std::find_if(queue.waiting_reqs.begin(), queue.waiting_reqs.end(),
                          [&rid](const auto& r) { return r->rid == rid; });
  if (wit != queue.waiting_reqs.end()) {
    queue.waiting_reqs.erase(wit);
  }
  queue.running_reqs.push_back(req);
}

void PrismRequestTracker::finish_req(const std::string& rid) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = all_reqs_.find(rid);
  if (it == all_reqs_.end()) {
    return;  // Already finished or unknown
  }
  auto& req = it->second;
  req->state = PrismReqState::FINISHED;
  req->finish_time = now_seconds();

  auto& queue = model_queues_[req->model];
  queue.last_active_time = now_seconds();

  // Remove from running
  queue.running_reqs.remove_if(
      [&rid](const auto& r) { return r->rid == rid; });

  // Remove from waiting (in case it was never started)
  auto wit = std::find_if(queue.waiting_reqs.begin(), queue.waiting_reqs.end(),
                          [&rid](const auto& r) { return r->rid == rid; });
  if (wit != queue.waiting_reqs.end()) {
    queue.waiting_reqs.erase(wit);
  }

  all_reqs_.erase(it);
}

std::unordered_map<std::string, PrismModelQueue>
PrismRequestTracker::snapshot_queues() {
  std::lock_guard<std::mutex> lock(mutex_);
  // Deep copy
  std::unordered_map<std::string, PrismModelQueue> result;
  for (const auto& [model, queue] : model_queues_) {
    PrismModelQueue copy;
    copy.model_name = queue.model_name;
    copy.last_active_time = queue.last_active_time;
    for (const auto& req : queue.waiting_reqs) {
      copy.waiting_reqs.push_back(std::make_shared<PrismReq>(*req));
    }
    for (const auto& req : queue.running_reqs) {
      copy.running_reqs.push_back(std::make_shared<PrismReq>(*req));
    }
    result[model] = std::move(copy);
  }
  return result;
}

int32_t PrismRequestTracker::get_total_reqs_on_instance(
    const std::string& instance_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  int32_t count = 0;
  for (const auto& [_, req] : all_reqs_) {
    if (req->instance_name == instance_name &&
        req->state == PrismReqState::RUNNING) {
      ++count;
    }
  }
  // Also count waiting requests (they don't have instance_name yet, but
  // we may want to count all active requests for the instance)
  return count;
}

double PrismRequestTracker::get_model_last_active_time(
    const std::string& model) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = model_queues_.find(model);
  if (it == model_queues_.end()) {
    return 0.0;
  }
  return it->second.last_active_time;
}

void PrismRequestTracker::update_time_budgets() {
  std::lock_guard<std::mutex> lock(mutex_);
  double now = now_seconds();
  for (auto& [_, req] : all_reqs_) {
    req->remaining_time_budget = req->slo - (now - req->arrival_time);
  }
}

void PrismRequestTracker::evict_waiting_reqs(const std::string& model) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = model_queues_.find(model);
  if (it == model_queues_.end()) return;

  // Remove all waiting requests from all_reqs_
  for (const auto& req : it->second.waiting_reqs) {
    all_reqs_.erase(req->rid);
  }
  int evicted = static_cast<int>(it->second.waiting_reqs.size());
  it->second.waiting_reqs.clear();

  if (evicted > 0) {
    LOG(INFO) << "Prism: evicted " << evicted
              << " waiting requests for model " << model;
  }
}

}  // namespace xllm_service
