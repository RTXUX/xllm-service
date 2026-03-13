/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm-service/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "instance_mgr.h"

#include <absl/strings/str_join.h>
#include <brpc/controller.h>
#include <glog/logging.h>

#include <chrono>
#include <thread>
#include <iostream>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <shared_mutex>

#include "scheduler/resource_model/linear_resource_model.h"

#include "common/global_gflags.h"
#include "common/types.h"
#include "common/utils.h"

namespace xllm_service {

static std::unordered_map<InstanceType, std::string> ETCD_KEYS_PREFIX_MAP = {
    {InstanceType::DEFAULT, "XLLM:DEFAULT:"},
    {InstanceType::PREFILL, "XLLM:PREFILL:"},
    {InstanceType::DECODE, "XLLM:DECODE:"},
    {InstanceType::MIX, "XLLM:MIX:"},
};
static std::string ETCD_ALL_KEYS_PREFIX = "XLLM:";
static std::string ETCD_LOADMETRICS_PREFIX = "XLLM:LOADMETRICS:";

InstanceMgr::InstanceMgr(const Options& options,
                         const std::shared_ptr<EtcdClient>& etcd_client,
                         const bool is_master_service)
    : options_(options),
      is_master_service_(is_master_service),
      etcd_client_(etcd_client) {
  auto handle_instance_metainfo =
      std::bind(&InstanceMgr::update_instance_metainfo,
                this,
                std::placeholders::_1,
                std::placeholders::_2);
  for (auto& it : ETCD_KEYS_PREFIX_MAP) {
    etcd_client_->add_watch(it.second, handle_instance_metainfo);
  }
  if (!is_master_service_) {
    auto handle_load_metrics = std::bind(&InstanceMgr::update_load_metrics,
                                         this,
                                         std::placeholders::_1,
                                         std::placeholders::_2);
    etcd_client_->add_watch(ETCD_LOADMETRICS_PREFIX, handle_load_metrics);
  }

  init();
}

void InstanceMgr::init() {
  init_model_memory_specs();
  init_model_resource_coefficients();

  {
    std::unique_lock<std::shared_mutex> mgr_lock(model_instance_mgr_mutex_);
    for (const auto& model_pair : MODELS) {
      model_instance_mgrs_.try_emplace(
          model_pair.first, std::make_shared<ModelInstanceMgr>(model_pair.first));
    }
  }

  // Start steady pool repack timer thread
  static constexpr int kRepackIntervalSeconds = 30;
  repack_thread_ = std::make_unique<std::thread>([this]() {
    while (!exited_) {
      std::this_thread::sleep_for(std::chrono::seconds(kRepackIntervalSeconds));
      if (exited_) break;
      steady_part_auto_repacking();
    }
  });

  // Start elastic-to-steady demotion check thread
  static constexpr int kDemotionCheckIntervalSeconds = 10;
  demotion_thread_ = std::make_unique<std::thread>([this]() {
    while (!exited_) {
      std::this_thread::sleep_for(
          std::chrono::seconds(kDemotionCheckIntervalSeconds));
      if (exited_) break;
      elastic_to_steady_demotion();
    }
  });

  {
    std::unique_lock<std::shared_mutex> lock(inst_mutex_);
    for (auto& it : ETCD_KEYS_PREFIX_MAP) {
      etcd_client_->get_prefix(it.second, &instances_);
    }
    // create ttft predictor and request metrics for each instance
    {
      std::lock_guard<std::mutex> time_predictor_lock(time_predictor_mutex_);
      std::lock_guard<std::mutex> request_metrics_lock(request_metrics_mutex_);
      for (auto& pair : instances_) {
        time_predictors_.insert_or_assign(
            pair.first,
            TimePredictor(pair.second.ttft_profiling_data,
                          pair.second.tpot_profiling_data));
        request_metrics_.insert_or_assign(pair.first, RequestMetrics());
      }
    }
    // Sync total_available_gpus_ with instances loaded from etcd, so that
    // subsequent deletes (which call fetch_sub) don't drive the counter negative.
    total_available_gpus_.store(
        static_cast<int32_t>(instances_.size()) * kTensorParallelSize);
    LOG(INFO) << "Load instance info from etcd:" << instances_.size();
    std::vector<std::string> channel_creat_fail_insts;
    for (auto& ist : instances_) {
      if (!create_channel(ist.first)) {
        channel_creat_fail_insts.emplace_back(ist.first);
      } else {
        std::unique_lock<std::shared_mutex> mgr_lock(model_instance_mgr_mutex_);
        // TODO: support multi-model registration, currently assuming instances serve all models or we need info from somewhere
        // For now, assuming instances serve all known models, or we need to parse from InstanceMetaInfo which model it serves
        // But InstanceMetaInfo struct doesn't have model list. Assuming homogenous cluster or we need to add model info to registration.
        // Assuming simple case: register to all ModelInstanceMgrs corresponding to served models.
        // But wait, InstanceMetaInfo structure is:
        /*
        struct InstanceMetaInfo {
            std::string name;
            std::string ip;
            int32_t port;
            InstanceType type;
            InstanceType current_type;
            int32_t instance_index;
            ...
        };
        */
        // It lacks model info. In multi-model world, we need to know which model an instance serves.
        // The prompt says "string model_id mapped to ModelInstanceMgr".
        // Assuming existing code structure, maybe we register instance to ALL model mgrs?
        // Or we need to look at how `get_next_instance_pair` uses `model_id`.
        // It seems `InstanceMgr` was serving one model type implicitly or mixing them up?
        // Ah, `get_next_instance_pair` takes `model_id`.
        // The original code had `prefill_index_` etc. which were GLOBAL for all models?
        // Yes, "InstanceMgr ... was only supporting one model type".
        // Now "multi-model".
        // We should iterate over supported models and create managers?
        // Or create manager on demand?
        
        // Let's create managers for all hardcoded MODELS for now.
        for (const auto& model_pair : MODELS) {
          std::string model_id = model_pair.first;
          if (model_instance_mgrs_.find(model_id) == model_instance_mgrs_.end()) {
             model_instance_mgrs_[model_id] = std::make_shared<ModelInstanceMgr>(model_id);
          }
          model_instance_mgrs_[model_id]->add_instance(ist.first, ist.second);
        }
      }
    }
    for (auto& name : channel_creat_fail_insts) {
      instances_.erase(name);
      {
        std::lock_guard<std::mutex> time_predictor_lock(time_predictor_mutex_);
        std::lock_guard<std::mutex> request_metrics_lock(
            request_metrics_mutex_);
        time_predictors_.erase(name);
        request_metrics_.erase(name);
      }
    }
  }
  {
    std::unique_lock<std::shared_mutex> lock(load_metric_mutex_);
    etcd_client_->get_prefix(ETCD_LOADMETRICS_PREFIX, &load_metrics_);
  }

}

InstanceMgr::~InstanceMgr() {
  exited_ = true;
  if (repack_thread_ && repack_thread_->joinable()) {
    repack_thread_->join();
  }
}

InstanceMetaInfo InstanceMgr::get_instance_info(
    const std::string& instance_name) {
  std::shared_lock<std::shared_mutex> lock(inst_mutex_);
  if (instances_.find(instance_name) == instances_.end()) {
    LOG(ERROR) << "Get instance info failed, instance is not registered, "
                  "instance_name: "
               << instance_name;
    return InstanceMetaInfo();
  }
  return instances_[instance_name];
}

bool InstanceMgr::get_next_instance_pair(const std::string& model_id, Routing* routing) {
  std::shared_lock<std::shared_mutex> lock(model_instance_mgr_mutex_);
  auto it = model_instance_mgrs_.find(model_id);
  if (it == model_instance_mgrs_.end()) {
    LOG(ERROR) << "Model manager not found for model " << model_id;
    return false;
  }
  return it->second->get_next_instance_pair(routing);
}

// TODO: refactor later, currently return all decode instances
std::vector<std::string> InstanceMgr::get_static_decode_list(
    const std::string& instance_name) {
  // Logic needs to find which model this instance serves or just return all?
  // Original code returned all DECODE instances.
  // With ModelInstanceMgr, we might need to query specific mgr?
  // But this method takes instance_name... unused?
  // Wait, argument is `instance_name` but not used in search?
  // "currently return all decode instances"
  
  // For now, let's keep it global if possible, OR we need to know the model context.
  // The caller likely wants peers.
  
  std::vector<std::string> decode_list;
  std::shared_lock<std::shared_mutex> lock(inst_mutex_);
  for (auto& inst : instances_) {
    if (inst.second.type == InstanceType::DECODE) {
      decode_list.emplace_back(inst.second.name);
    }
  }
  return decode_list;
}

// TODO: refactor later, currently return all prefill instances
std::vector<std::string> InstanceMgr::get_static_prefill_list(
    const std::string& instance_name) {
  std::vector<std::string> prefill_list;
  std::shared_lock<std::shared_mutex> lock(inst_mutex_);
  for (auto& inst : instances_) {
    if (inst.second.type == InstanceType::PREFILL ||
        inst.second.type == InstanceType::DEFAULT) {
      prefill_list.emplace_back(inst.second.name);
    }
  }
  return prefill_list;
}

void InstanceMgr::fork_master_and_sleep(
    const std::string& instance_name,
    std::shared_ptr<brpc::Channel> channel) {
  for (const auto& model : MODELS) {
    // 1. Fork Master

    nlohmann::json fork_body;
    fork_body["model_path"] = model.second;
    fork_body["master_node_addr"] = "127.0.0.1:" + std::to_string(++master_node_port);
    fork_body["master_status"] = 1;
    fork_body["nnodes"] = kTensorParallelSize;

    auto model_id = model.first;

    /* hardcoded for now */
    int base_port = stoi(instance_name.substr(instance_name.find(":") + 1));

    std::vector<std::thread> fork_threads;
    std::atomic<int> fork_success_count(0);

    for (int node_idx = 0; node_idx < kTensorParallelSize; ++node_idx) {
      
      /* hardcoded for now */
      int tmp_port = base_port + node_idx;
      std::string tmp_instance_name = instance_name.substr(0, instance_name.find(":")) +
                                      ":" + std::to_string(tmp_port);
      std::shared_ptr<brpc::Channel> tmp_channel = cached_channels_[tmp_instance_name];


      fork_threads.emplace_back([this, tmp_instance_name, node_idx, fork_body, model_id, tmp_channel, &fork_success_count]() {
        // send fork request

        if (node_idx > 0) {
          std::this_thread::sleep_for(std::chrono::seconds(2));
        }

        for (int i = 0; i < 10; ++i) {
          if (send_http_request(tmp_channel, "/fork_master", fork_body.dump())) {
            fork_success_count += 1;
            break;
          }
          LOG(WARNING) << "Failed to fork master for model " << model_id << " on "
                       << tmp_instance_name << ", retry " << i + 1;
          std::this_thread::sleep_for(std::chrono::seconds(1));
        }      

      });

    }

    for (auto& t : fork_threads) {
      if (t.joinable()) {
        t.join();
      }
    }

    if (fork_success_count.load() != kTensorParallelSize) {
      LOG(ERROR) << "Failed to fork master for model " << model.first << " on "
                << instance_name << " after retries";
      continue;
    }

    auto mgr = get_model_instance_mgr(model.first);
    mgr->set_model_state(instance_name, ModelState::SLEEP);

    // 2. force sleep (the initial model of xllm instance fails to fork_master)
    nlohmann::json sleep_body;
    sleep_body["model_id"] = model.first;
    sleep_body["master_status"] = 1;
    
    send_http_request(channel, "/sleep", sleep_body.dump());
  }

  // All models fork_master'd — bidirectional D2D linking with all ready instances
  {
    auto new_info = get_instance_xtensor_info(instance_name);
    if (new_info && !new_info->device_addrs.empty()) {
      // Snapshot fork_done instances (release lock before acquiring xtensor_info_mutex_)
      std::vector<std::string> done_peers;
      {
        std::lock_guard<std::mutex> lock(fork_done_mutex_);
        done_peers.assign(fork_done_instances_.begin(),
                          fork_done_instances_.end());
      }

      // Gather peer channels and device addrs
      std::vector<std::pair<std::shared_ptr<brpc::Channel>,
                             std::vector<std::string>>> peers;
      {
        std::lock_guard<std::mutex> xt_lock(xtensor_info_mutex_);
        for (const auto& peer_name : done_peers) {
          auto it = instance_xtensor_infos_.find(peer_name);
          if (it != instance_xtensor_infos_.end() && it->second.is_valid &&
              !it->second.device_addrs.empty()) {
            auto peer_channel = get_channel(peer_name);
            if (peer_channel) {
              peers.emplace_back(peer_channel, it->second.device_addrs);
            }
          }
        }
      }

      // Use first model's mgr for the link call (mooncake session is model-agnostic)
      auto mgr = get_model_instance_mgr(MODELS[0].first);
      if (mgr) {
        mgr->link_d2d_bidirectional(channel, new_info->device_addrs, peers);
      }

      // Mark self as fork_done
      {
        std::lock_guard<std::mutex> lock(fork_done_mutex_);
        fork_done_instances_.insert(instance_name);
      }
      LOG(INFO) << "Instance " << instance_name
                << " completed all model forks and D2D linking";
    } else {
      LOG(WARNING) << "No device addrs for instance " << instance_name
                   << ", skipping D2D linking";
      // Still mark as fork_done so others can link to us later
      std::lock_guard<std::mutex> lock(fork_done_mutex_);
      fork_done_instances_.insert(instance_name);
    }
  }
}

bool InstanceMgr::send_http_request(const std::string& instance_name,
                                    const std::string& uri,
                                    const std::string& request_body) {
  std::shared_ptr<brpc::Channel> channel = get_channel(instance_name);
  if (!channel) {
    LOG(ERROR) << "Channel not found for " << instance_name;
    return false;
  }
  return send_http_request(channel, uri, request_body);
}

bool InstanceMgr::send_http_request(std::shared_ptr<brpc::Channel> channel,
                                    const std::string& uri,
                                    const std::string& request_body) {
  brpc::Controller cntl;
  cntl.http_request().uri() = uri;  // brpc channel already has host:port
  cntl.http_request().set_method(brpc::HTTP_METHOD_POST);
  cntl.http_request().set_content_type("application/json");
  cntl.request_attachment().append(request_body);

  channel->CallMethod(nullptr, &cntl, nullptr, nullptr, nullptr);

  if (cntl.Failed()) {
    LOG(ERROR) << "HTTP request failed: " << cntl.ErrorText();
    return false;
  }
  return true;
}

std::optional<LoadMetrics> InstanceMgr::get_instance_load_metrics(
    const std::string& instance_name) {
  std::shared_lock<std::shared_mutex> lock(load_metric_mutex_);
  auto it = load_metrics_.find(instance_name);
  if (it == load_metrics_.end()) return std::nullopt;
  return it->second;
}

void InstanceMgr::get_load_metrics(LoadBalanceInfos* infos) {
  std::shared_lock<std::shared_mutex> inst_lock(inst_mutex_);
  std::shared_lock<std::shared_mutex> metric_lock(load_metric_mutex_);

  for (auto name : infos->overlap_scores.instances) {
    auto it = load_metrics_.find(name);
    if (it == load_metrics_.end()) {
      continue;
    }
    auto instance_it = instances_.find(name);
    if (instance_it == instances_.end()) {
      continue;
    }

    if (instance_it->second.type == InstanceType::DECODE) {
      infos->decode_load_metrics.insert(std::make_pair(name, it->second));
      infos->decode_max_waiting_requests_num =
          std::max(infos->decode_max_waiting_requests_num,
                   it->second.waiting_requests_num);
    } else {
      infos->prefill_load_metrics.insert(std::make_pair(name, it->second));
      infos->prefill_max_waiting_requests_num =
          std::max(infos->prefill_max_waiting_requests_num,
                   it->second.waiting_requests_num);
    }
  }

  std::string least_loaded_prefill_instance;
  float least_loaded_prefill_gpu_cache_usage_perc = 1;
  std::string least_loaded_decode_instance;
  float least_loaded_decode_gpu_cache_usage_perc = 1;

  if (infos->prefill_load_metrics.size() == 0 ||
      infos->decode_load_metrics.size() == 0) {
    for (const auto& metric : load_metrics_) {
      auto instance_it = instances_.find(metric.first);
      if (instance_it != instances_.end()) {
        if (instance_it->second.type != InstanceType::DECODE) {
          if (metric.second.gpu_cache_usage_perc <
              least_loaded_prefill_gpu_cache_usage_perc) {
            least_loaded_prefill_gpu_cache_usage_perc =
                metric.second.gpu_cache_usage_perc;
            least_loaded_prefill_instance = metric.first;
          }
        } else {
          if (metric.second.gpu_cache_usage_perc <
              least_loaded_decode_gpu_cache_usage_perc) {
            least_loaded_decode_gpu_cache_usage_perc =
                metric.second.gpu_cache_usage_perc;
            least_loaded_decode_instance = metric.first;
          }
        }
      }
    }
  }

  if (infos->prefill_load_metrics.size() == 0 &&
      !least_loaded_prefill_instance.empty()) {
    infos->prefill_load_metrics.insert(
        std::make_pair(least_loaded_prefill_instance,
                       load_metrics_[least_loaded_prefill_instance]));
  }

  if (infos->decode_load_metrics.size() == 0 &&
      !least_loaded_decode_instance.empty()) {
    infos->decode_load_metrics.insert(
        std::make_pair(least_loaded_decode_instance,
                       load_metrics_[least_loaded_decode_instance]));
  }
}

void InstanceMgr::record_load_metrics_update(
    const std::string& instance_name,
    const proto::LoadMetrics& load_metrics) {
  std::lock_guard<std::mutex> lock(update_mutex_);

  updated_metrics_.insert_or_assign(
      instance_name,
      LoadMetrics(load_metrics.waiting_requests_num(),
                  load_metrics.gpu_cache_usage_perc()));
}

bool InstanceMgr::upload_load_metrics() {
  std::lock_guard<std::mutex> lock(update_mutex_);
  bool status = etcd_client_->set(ETCD_LOADMETRICS_PREFIX, updated_metrics_);
  status =
      status && etcd_client_->rm(ETCD_LOADMETRICS_PREFIX, removed_instance_);
  {
    std::unique_lock<std::shared_mutex> lock(inst_mutex_);
    for (auto& iter : updated_metrics_) {
      load_metrics_.insert_or_assign(iter.first, std::move(iter.second));
    }
    for (auto& iter : removed_instance_) {
      load_metrics_.erase(iter);
    }
  }
  updated_metrics_.clear();
  removed_instance_.clear();

  return status;
}

void InstanceMgr::set_as_master() {
  is_master_service_ = true;
  etcd_client_->remove_watch(ETCD_LOADMETRICS_PREFIX);
}

void InstanceMgr::on_heartbeat(const std::string& instance_name) {
  InstanceMetaInfo metainfo;
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    auto it = pending_infos_.find(instance_name);
    if (it == pending_infos_.end()) {
      return;
    }
    metainfo = std::move(it->second);
    pending_infos_.erase(it);
  }

  LOG(INFO) << "Received heartbeat from pending instance: " << instance_name;
  threadpool_.schedule([this, instance_name, metainfo = std::move(metainfo)]() {
    register_instance(instance_name, metainfo);
  });
}

void InstanceMgr::register_instance(const std::string& instance_name,
                                    InstanceMetaInfo metainfo) {
  std::unique_lock<std::shared_mutex> lock(inst_mutex_);
  if (instances_.find(instance_name) != instances_.end()) {
    LOG(ERROR) << "Instance is already registered, instance_name: "
               << instance_name;
    return;
  }
  
  if (!create_channel(instance_name)) {
    LOG(ERROR) << "create channel fail: " << instance_name;
    return;
  }

  /* hardcoded for now*/
  if (kTensorParallelSize > 1) {
    for (int node_idx = 1; node_idx < kTensorParallelSize; ++node_idx) {
      int instance_port = stoi(instance_name.substr(instance_name.find(":") + 1));
      int tmp_port = instance_port + node_idx;
      std::string tmp_instance_name = instance_name.substr(0, instance_name.find(":")) +
                                      ":" + std::to_string(tmp_port);
      if (!create_channel(tmp_instance_name)) {
        LOG(ERROR) << "create channel fail: " << tmp_instance_name;
        return;
      }
    }
  }
  
  // Note: we can't call fork_master_and_sleep here if we are holding
  // inst_mutex_ and fork_master_and_sleep calls send_http_request which calls
  // get_channel which acquires inst_mutex_ again (deadlock).
  auto channel = cached_channels_[instance_name];
  threadpool_.schedule([this, instance_name, channel]() {
    fork_master_and_sleep(instance_name, channel);
  });

  // Initialize xtensor info as invalid only if not already set
  // (update_xtensor_info from the same heartbeat may have already set a valid entry)
  {
    std::lock_guard<std::mutex> lock(xtensor_info_mutex_);
    if (instance_xtensor_infos_.find(instance_name) == instance_xtensor_infos_.end()) {
      InstanceXTensorInfo info;
      info.is_valid = false;  // Mark as not yet received valid data
      // Populate device_addrs from registration info for early D2D linking
      info.device_addrs = metainfo.device_addrs;
      info.p2p_addrs = metainfo.p2p_addrs;
      instance_xtensor_infos_[instance_name] = std::move(info);
    } else if (instance_xtensor_infos_[instance_name].device_addrs.empty()) {
      // Entry exists but missing device_addrs — fill from registration
      instance_xtensor_infos_[instance_name].device_addrs = metainfo.device_addrs;
      instance_xtensor_infos_[instance_name].p2p_addrs = metainfo.p2p_addrs;
    }
  }

  {
    std::lock_guard<std::mutex> time_predictor_lock(time_predictor_mutex_);
    std::lock_guard<std::mutex> request_metrics_lock(request_metrics_mutex_);
    // create ttft predictor for instance
    time_predictors_.emplace(instance_name,
                             TimePredictor(metainfo.ttft_profiling_data,
                                           metainfo.tpot_profiling_data));

    // create request metrics for instance
    request_metrics_.emplace(instance_name, RequestMetrics());
    for (auto& model : MODELS) {
      request_metrics_[instance_name].model_metrics.try_emplace(model.first);
    }
  }

  // Register with ModelInstanceMgrs
  {
    std::unique_lock<std::shared_mutex> mgr_lock(model_instance_mgr_mutex_);
    for (const auto& model_pair : MODELS) {
      std::string model_id = model_pair.first;
      if (model_instance_mgrs_.find(model_id) == model_instance_mgrs_.end()) {
          model_instance_mgrs_[model_id] = std::make_shared<ModelInstanceMgr>(model_id);
      }
      model_instance_mgrs_[model_id]->add_instance(instance_name, metainfo);
    }
  }

  // Legacy code cleanup / adaptation
  // Note: original code updated indices here. ModelInstanceMgr does it internally.
  // We still need to respect InstanceType logging if needed.
  
  /*
  switch (metainfo.type) {
    // ...
  }
  */
  // Since we moved logic to ModelInstanceMgr, we might just log here.
  LOG(INFO) << "Registered instance " << instance_name << " type " << (int)metainfo.type;

  instances_.insert(std::make_pair(instance_name, std::move(metainfo)));
  total_available_gpus_.fetch_add(kTensorParallelSize);
}

std::shared_ptr<brpc::Channel> InstanceMgr::get_channel(
    const std::string& instance_name) {
  std::shared_lock<std::shared_mutex> lock(inst_mutex_);
  auto iter = cached_channels_.find(instance_name);
  if (iter == cached_channels_.end()) {
    return nullptr;
  }
  return iter->second;
}

bool InstanceMgr::create_channel(const std::string& instance_name) {
  if (cached_channels_.find(instance_name) == cached_channels_.end()) {
    auto channel = std::make_shared<brpc::Channel>();
    brpc::ChannelOptions options;
    // Add to params
    options.protocol = "http";
    options.timeout_ms = options_.timeout_ms(); /*milliseconds*/
    options.max_retry = 3;
    std::string load_balancer = "";
    if (channel->Init(instance_name.c_str(), load_balancer.c_str(), &options) !=
        0) {
      LOG(ERROR) << "Fail to initialize channel for " << instance_name;
      return false;
    }
    cached_channels_[instance_name] = std::move(channel);
  }

  return true;
}

void InstanceMgr::update_instance_metainfo(const etcd::Response& response,
                                           const uint64_t& prefix_len) {
  if (response.events().empty() || exited_) {
    return;
  }

  threadpool_.schedule([this,
                        response = std::move(response),
                        prefix_len = std::move(prefix_len)] {
    if (exited_) return;
    std::unordered_map<std::string, InstanceMetaInfo> put_map;
    std::vector<std::string> delete_list;

    for (const auto& event : response.events()) {
      std::string instance_name = event.kv().key().substr(prefix_len);

      if (event.event_type() == etcd::Event::EventType::PUT) {
        InstanceMetaInfo metainfo;
        auto json_str = event.kv().as_string();
        if (!metainfo.parse_from_json(json_str)) {
          LOG(ERROR) << "pase json:" << json_str << " error!";
          continue;
        }
        put_map.insert(std::make_pair(instance_name, std::move(metainfo)));

      } else if (event.event_type() == etcd::Event::EventType::DELETE_) {
        delete_list.push_back(instance_name);
      }
    }

    {
      std::unique_lock<std::shared_mutex> lock(inst_mutex_);
      for (auto& iter : put_map) {
        if (instances_.find(iter.first) != instances_.end()) {
          // Update existing instance profiling data
          LOG(INFO) << "Update instance profiling data, instance_name: " << iter.first;
          auto& exist_info = instances_[iter.first];
          auto& new_info = iter.second;
          
          // Merge TTFT profiling data
          for (auto& [model_id, data] : new_info.ttft_profiling_data) {
            exist_info.ttft_profiling_data[model_id] = std::move(data);
          }
          
          // Merge TPOT profiling data
          for (auto& [model_id, data] : new_info.tpot_profiling_data) {
            exist_info.tpot_profiling_data[model_id] = std::move(data);
          }

          // Update TimePredictor
          {
            std::lock_guard<std::mutex> time_predictor_lock(time_predictor_mutex_);
            time_predictors_.insert_or_assign(
                iter.first,
                TimePredictor(exist_info.ttft_profiling_data,
                              exist_info.tpot_profiling_data));
          }
          continue;
        }

        {
          std::lock_guard<std::mutex> lock(pending_mutex_);
          if (pending_infos_.count(iter.first)) {
            LOG(INFO) << "Instance is pending, instance_name: " << iter.first;
            continue;
          }
          pending_infos_.insert(
              std::make_pair(iter.first, std::move(iter.second)));
          LOG(INFO) << "Add instance to pending list and wait for heartbeat: "
                    << iter.first;
        }
      }

      for (auto& iter : delete_list) {
        LOG(INFO) << "delete instance: " << iter;
        {
          std::lock_guard<std::mutex> lock(pending_mutex_);
          if (pending_infos_.count(iter)) {
            pending_infos_.erase(iter);
            LOG(INFO) << "Delete pending instance: " << iter;
            continue;
          }
        }
        if (instances_.find(iter) == instances_.end()) {
          LOG(ERROR) << "Instance is already deleted, instance_name: " << iter;
          continue;
        }
        // TODO: notify cache manager to clear expire cache
        // Remove from ModelInstanceMgrs
        {
          std::unique_lock<std::shared_mutex> mgr_lock(model_instance_mgr_mutex_);
          for (auto& pair : model_instance_mgrs_) {
            pair.second->remove_instance(iter);
          }
        }

        instances_.erase(iter);
        cached_channels_.erase(iter);
        total_available_gpus_.fetch_sub(kTensorParallelSize);
        {
          std::lock_guard<std::mutex> time_predictor_lock(
              time_predictor_mutex_);
          std::lock_guard<std::mutex> request_metrics_lock(
              request_metrics_mutex_);
          time_predictors_.erase(iter);
          request_metrics_.erase(iter);
        }
        {
          std::lock_guard<std::mutex> lock(update_mutex_);
          updated_metrics_.erase(iter);
          removed_instance_.insert(iter);
        }
      }
    }
  });
}

void InstanceMgr::update_load_metrics(const etcd::Response& response,
                                      const uint64_t& prefix_len) {
  if (response.events().empty() || exited_) {
    return;
  }
  threadpool_.schedule([this,
                        response = std::move(response),
                        prefix_len = std::move(prefix_len)] {
    if (exited_) return;
    std::unordered_map<std::string, LoadMetrics> put_map;
    std::vector<std::string> delete_list;

    for (const auto& event : response.events()) {
      std::string instance_name = event.kv().key().substr(prefix_len);

      if (event.event_type() == etcd::Event::EventType::PUT) {
        LoadMetrics load_metrics;
        auto json_str = event.kv().as_string();
        if (!load_metrics.parse_from_json(json_str)) {
          LOG(ERROR) << "pase json:" << json_str << " error!";
          continue;
        }

        put_map.insert(std::make_pair(instance_name, std::move(load_metrics)));

      } else if (event.event_type() == etcd::Event::EventType::DELETE_) {
        delete_list.push_back(instance_name);
      }
    }

    {
      std::unique_lock<std::shared_mutex> lock(load_metric_mutex_);
      for (auto& iter : put_map) {
        load_metrics_.insert_or_assign(iter.first, std::move(iter.second));
      }

      for (auto& iter : delete_list) {
        load_metrics_.erase(iter);
      }
    }
  });
}

void InstanceMgr::update_latency_metrics(
    const std::string& instance_name,
    const proto::LatencyMetrics& latency_metrics) {
  std::lock_guard<std::mutex> lock(latency_metrics_mutex_);

  LatencyMetrics metrics;
  for (const auto& entry : latency_metrics.model_metrics()) {
    const std::string& model_id = entry.first;
    const auto& proto_model_metrics = entry.second;

    ModelLatencyMetrics model_metrics;
    model_metrics.recent_max_ttft = proto_model_metrics.recent_max_ttft();
    model_metrics.recent_max_tbt = proto_model_metrics.recent_max_tbt();
    
    metrics.model_metrics[model_id] = model_metrics;
  }
  
  latency_metrics_.insert_or_assign(instance_name, std::move(metrics));
}

void InstanceMgr::update_request_metrics(std::shared_ptr<Request> request,
                                         RequestAction action) {
  std::lock_guard<std::mutex> lock(request_metrics_mutex_);

  auto prefill_instance_it = request_metrics_.find(request->routing.prefill_name);
  if (prefill_instance_it == request_metrics_.end()) {
    LOG(ERROR) << "Failed to find prefill instance request metrics, instance name : "
               << request->routing.prefill_name;
    return;
  }

  auto prefill_model_it = prefill_instance_it->second.model_metrics.find(
      request->model);
  if (prefill_model_it == prefill_instance_it->second.model_metrics.end()) {
    LOG(ERROR) << "Failed to find prefill model request metrics, instance name : "
               << request->routing.prefill_name
               << ", model id : " << request->model;
    return;
  }

  if (request->routing.decode_name.empty()) {
    request->routing.decode_name = request->routing.prefill_name;
  }

  auto decode_instance_it = request_metrics_.find(request->routing.decode_name);
  if (decode_instance_it == request_metrics_.end()) {
    LOG(ERROR) << "Failed to find decode instance request metrics, instance name : "
               << request->routing.decode_name;
    return;
  }

  auto decode_model_it = decode_instance_it->second.model_metrics.find(
      request->model);
  if (decode_model_it == decode_instance_it->second.model_metrics.end()) {
    LOG(ERROR) << "Failed to find decode model request metrics, instance name : "
               << request->routing.decode_name
               << ", model id : " << request->model;
    return;
  }

  int64_t num_prompt_tokens = request->token_ids.size();
  int64_t num_generated_tokens = request->num_generated_tokens;
  switch (action) {
    case RequestAction::SCHEDULE:
      // update the request metrics for prefill and decode instances when
      // request is scheduled
      prefill_model_it->second.prefill_request_num += 1;
      prefill_model_it->second.prefill_token_num += num_prompt_tokens;

      decode_model_it->second.decode_request_num += 1;
      decode_model_it->second.decode_token_num += num_prompt_tokens;

      // Update estimated_prefill_done_time if not already set by SLO_AWARE selection
      if (request->expected_prefill_done_ms == 0) {
        int64_t now_ms = absl::ToUnixMillis(absl::Now());
        auto& epdt = prefill_instance_it->second.estimated_prefill_done_time;
        epdt = std::max(epdt, now_ms) + request->estimated_ttft;
        request->expected_prefill_done_ms = epdt;
      }

      // Track this request for EPDT correction propagation
      inflight_prefill_requests_[request->routing.prefill_name].push_back(request);
      break;
    case RequestAction::FINISH_PREFILL:
      // update the request metrics for prefill and decode instance when request
      // finishes the prefill phase
      prefill_model_it->second.prefill_request_num -= 1;
      prefill_model_it->second.prefill_token_num -= num_prompt_tokens;

      // Apply correction feedback to estimated_prefill_done_time
      {
        int64_t now_ms = absl::ToUnixMillis(absl::Now());
        auto& epdt = prefill_instance_it->second.estimated_prefill_done_time;
        if (request->expected_prefill_done_ms > 0) {
          // Shift EPDT by the difference between actual and expected completion
          int64_t correction = now_ms - request->expected_prefill_done_ms;
          epdt += correction;

          // Propagate correction to all remaining inflight requests on this
          // instance so their future FINISH_PREFILL won't double-count.
          auto& inflight =
              inflight_prefill_requests_[request->routing.prefill_name];
          for (auto& req : inflight) {
            if (req->service_request_id != request->service_request_id) {
              req->expected_prefill_done_ms += correction;
            }
          }
        }
        // Clamp: EPDT should never be in the past
        epdt = std::max(epdt, now_ms);

        // Remove this request from inflight list
        auto& inflight =
            inflight_prefill_requests_[request->routing.prefill_name];
        inflight.erase(
            std::remove_if(
                inflight.begin(), inflight.end(),
                [&](const std::shared_ptr<Request>& r) {
                  return r->service_request_id ==
                         request->service_request_id;
                }),
            inflight.end());
      }

      decode_model_it->second.decode_token_num += 1;
      break;
    case RequestAction::GENERATE:
      // update the request metrics for decode instance when request generate a
      // token
      decode_model_it->second.decode_token_num += 1;
      break;
    case RequestAction::FINISH_DECODE:
      // update the request metrics for decode instance when request finishes
      // the decode phase
      decode_model_it->second.decode_request_num -= 1;
      decode_model_it->second.decode_token_num -=
          (num_prompt_tokens + num_generated_tokens);
      break;
    case RequestAction::CANCEL:
      // update the request metrics for prefill and decode instances when
      // request is cancelled
      prefill_model_it->second.prefill_request_num -= 1;
      prefill_model_it->second.prefill_token_num -= num_prompt_tokens;

      // Remove this request's contribution from EPDT
      {
        auto& epdt = prefill_instance_it->second.estimated_prefill_done_time;
        int64_t removed_ttft = request->estimated_ttft;
        epdt -= removed_ttft;
        int64_t now_ms = absl::ToUnixMillis(absl::Now());
        // Clamp: EPDT should never be in the past
        epdt = std::max(epdt, now_ms);

        // Propagate removal to remaining inflight requests: their expected
        // times shift earlier by the cancelled request's contribution.
        auto& inflight =
            inflight_prefill_requests_[request->routing.prefill_name];
        for (auto& req : inflight) {
          if (req->service_request_id != request->service_request_id) {
            req->expected_prefill_done_ms -= removed_ttft;
          }
        }

        // Remove this request from inflight list
        inflight.erase(
            std::remove_if(
                inflight.begin(), inflight.end(),
                [&](const std::shared_ptr<Request>& r) {
                  return r->service_request_id ==
                         request->service_request_id;
                }),
            inflight.end());
      }

      decode_model_it->second.decode_request_num -= 1;
      decode_model_it->second.decode_token_num -=
          (num_prompt_tokens + num_generated_tokens);
      break;
    default:
      LOG(ERROR) << "Unknown RequestAction: " << static_cast<int32_t>(action);
      break;
  }

  if (action == RequestAction::FINISH_PREFILL ||
      action == RequestAction::FINISH_DECODE ||
      action == RequestAction::CANCEL) {

    if (prefill_model_it->second.prefill_request_num == 0 &&
        prefill_model_it->second.decode_request_num == 0) {
      prefill_model_it->second.cv_idle.notify_all();
    }

    if (request->routing.prefill_name != request->routing.decode_name &&
        decode_model_it->second.prefill_request_num == 0 &&
        decode_model_it->second.decode_request_num == 0) {
      decode_model_it->second.cv_idle.notify_all();
    }

  }

  /*
  if (options_.load_balance_policy() == "SLO_AWARE" &&
      decode_it->second.decode_request_num == 0) {
    std::unique_lock<std::shared_mutex> instance_lock(inst_mutex_);
    flip_decode_to_prefill(request->routing.decode_name);
  }
  */
}

bool InstanceMgr::select_instance_pair_on_slo(
    std::shared_ptr<Request> request) {
  std::unique_lock<std::shared_mutex> lock(inst_mutex_);
  std::lock_guard<std::mutex> request_metrics_lock(request_metrics_mutex_);
  auto awake_instances = get_awake_instances(request->model);
  if (awake_instances.empty()) {
    LOG(ERROR) << "No awake instance found for model " << request->model;
    return false;
  }

  // get earliest estimated_prefill_done_time instance from request metrics
  auto best_instance = awake_instances[0];
  int64_t min_done_time = std::numeric_limits<int64_t>::max();
  for (auto& instance : awake_instances) {
    int64_t done_time = request_metrics_[instance].estimated_prefill_done_time;
    if (done_time < min_done_time) {
      best_instance = instance;
      min_done_time = done_time;
    }
  }

  request->routing.prefill_name = best_instance;
  request->routing.decode_name = best_instance;
  request->estimated_ttft =
      predict_ttft(best_instance, request->model, request->token_ids.size());

  // Update EPDT immediately to prevent concurrent selections picking same instance
  {
    int64_t now_ms = absl::ToUnixMillis(absl::Now());
    auto& epdt = request_metrics_[best_instance].estimated_prefill_done_time;
    epdt = std::max(epdt, now_ms) + request->estimated_ttft;
    request->expected_prefill_done_ms = epdt;
  }

  return true;
}

int64_t InstanceMgr::get_estimated_prefill_done_time(
    const std::string& instance_name) {
  std::lock_guard<std::mutex> lock(request_metrics_mutex_);
  auto it = request_metrics_.find(instance_name);
  if (it != request_metrics_.end()) {
    return it->second.estimated_prefill_done_time;
  }
  return 0;
}

// flip all models
void InstanceMgr::flip_prefill_to_decode(std::string& instance_name) {
  {
    std::shared_lock<std::shared_mutex> mgr_lock(model_instance_mgr_mutex_);
    // Flip in all managers
    for (auto& pair : model_instance_mgrs_) {
      pair.second->flip_prefill_to_decode(instance_name);
    }
  }
  
  std::unique_lock<std::shared_mutex> inst_lock(inst_mutex_);
  if (instances_.find(instance_name) == instances_.end()) {
    LOG(ERROR) << "Can't find instance, instance_name: " << instance_name;
    return;
  }
  instances_[instance_name].current_type = InstanceType::DECODE;
  LOG(INFO) << "Flip prefill to decode, instance name : " << instance_name;
}

// flip all models
void InstanceMgr::flip_decode_to_prefill(std::string& instance_name) {
  {
    std::shared_lock<std::shared_mutex> mgr_lock(model_instance_mgr_mutex_);
    // Flip in all managers
    for (auto& pair : model_instance_mgrs_) {
      pair.second->flip_decode_to_prefill(instance_name);
    }
  }
  
  std::unique_lock<std::shared_mutex> inst_lock(inst_mutex_);
  if (instances_.find(instance_name) == instances_.end()) {
    LOG(ERROR) << "Can't find instance, instance_name: " << instance_name;
    return;
  }
  instances_[instance_name].current_type = InstanceType::PREFILL;
  LOG(INFO) << "Flip decode to prefill, instance name : " << instance_name;
}

TimePredictor& InstanceMgr::get_time_predictor(
    const std::string& instance_name) {
  std::lock_guard<std::mutex> lock(time_predictor_mutex_);

  auto it = time_predictors_.find(instance_name);
  if (it == time_predictors_.end()) {
    LOG(FATAL) << "Find TimePredictor failed, instance name : "
               << instance_name;
  }
  return it->second;
}

double InstanceMgr::predict_ttft(const std::string& instance_name,
                                  const std::string& model_id,
                                  int32_t token_count) {
  std::lock_guard<std::mutex> lock(time_predictor_mutex_);
  auto it = time_predictors_.find(instance_name);
  if (it != time_predictors_.end()) {
    double ttft = it->second.predict_ttft(model_id, token_count);
    if (ttft > 0) {
      return ttft;
    }
  }
  // Fallback: ttft_ms = 0.0678 * tokens + 19.63 (fitted from profiling)
  return static_cast<double>(token_count) * 0.0678 + 19.63;
}

double InstanceMgr::predict_ttft_any_instance(const std::string& model_id,
                                               int32_t token_count) {
  std::lock_guard<std::mutex> lock(time_predictor_mutex_);
  for (auto& [instance_name, predictor] : time_predictors_) {
    double ttft = predictor.predict_ttft(model_id, token_count);
    if (ttft > 0) {
      return ttft;
    }
  }
  // Fallback: ttft_ms = 0.0678 * tokens + 19.63 (fitted from profiling)
  return static_cast<double>(token_count) * 0.0678 + 19.63;
}

void InstanceMgr::send_model_sleep(const std::string& instance_name,
                                   const std::string& model_id) {
  if (instance_name.empty() || instance_name == "all") {
    LOG(ERROR) << "Only support fixed instance_name for model trigger now.";
    return;
  }

  auto model_mgr = get_model_instance_mgr(model_id);

  std::shared_ptr<brpc::Channel> channel = get_channel(instance_name);

  if (model_mgr->send_model_sleep(instance_name, channel)) {
    LOG(INFO) << "Model " << model_id << " on " << instance_name
              << " sleep successful. Memory freed will be reflected in next heartbeat.";
  }
}

void InstanceMgr::send_model_wakeup(const std::string& instance_name,
                                    const std::string& model_id,
                                    bool memory_increased_in_advance) {

  if (instance_name.empty() || instance_name == "all") {
    LOG(ERROR) << "Only support fixed instance_name for model trigger now.";
    return;
  }

  auto model_mgr = get_model_instance_mgr(model_id);

  std::shared_ptr<brpc::Channel> channel = get_channel(instance_name);

  // Try to find a D2D source instance (also acquires D2D lock if found)
  auto d2d_info = find_d2d_source(model_id, instance_name);

  bool wakeup_success = false;
  if (d2d_info.has_value()) {
    // Use D2D wakeup (mooncake connections already established at fork_master time)
    wakeup_success = model_mgr->send_model_wakeup_d2d(instance_name, channel, d2d_info.value());

    // Release D2D lock on the source instance (whether success or failure)
    model_mgr->release_d2d_lock(d2d_info->source_instance_name);

    if (!wakeup_success) {
      LOG(WARNING) << "D2D wakeup failed for model " << model_id
                   << " on " << instance_name << ", falling back to H2D";
      // D2D failure sets state to SLEEP, reset to ALLOCATED for H2D fallback
      model_mgr->set_model_state(instance_name, ModelState::ALLOCATED);
      // Fallback to H2D
      wakeup_success = model_mgr->send_model_wakeup(instance_name, channel);
    }
  } else {
    // Use H2D wakeup
    wakeup_success = model_mgr->send_model_wakeup(instance_name, channel);
  }

  if (wakeup_success) {
    LOG(INFO) << "Model " << model_id << " wakeup successful on " << instance_name
              << ". Memory usage will be reflected in next heartbeat.";
  } else {
    LOG(ERROR) << "Failed to wakeup model " << model_id
               << " on " << instance_name;
  }
}

void InstanceMgr::init_model_memory_specs() {
    // Hardcoded memory specs: x * 2 + 5 GB. 5GB = 3GB KV cache + 2GB overhead

    // just for test
    model_memory_specs_["Qwen3-4B"] = 25.0;
    model_memory_specs_["Qwen2-7B"] = 25.0;
    model_memory_specs_["Qwen3-8B"] = 25.0;
    

    // model_memory_specs_["Qwen3-8B"] = 8.0 * 2 + 5.0;// 21.0GB
    // model_memory_specs_["Qwen2-7B"] = 7.0 * 2 + 5.0;// 19.0GB
    // model_memory_specs_["Qwen2-7B-Instruct"] = 7.0 * 2 + 5.0;// 19.0GB
    // model_memory_specs_["Qwen2.5-14B"] = 14.0 * 2 + 5.0;// 33.0GB
    // model_memory_specs_["Qwen3-4B"] = 4.0 * 2 + 5.0;// 13.0GB
    // model_memory_specs_["Qwen2.5-3b"] = 3.0 * 2 + 5.0;// 11.0GB
    // model_memory_specs_["Qwen3-30B-A3B-Instruct-2507"] = 57.0 + 5.0;// 62.0GB
    // model_memory_specs_["Qwen3-30B-A3B-W8A8"] = 30.0 + 5.0;// 35.0GB
    // model_memory_specs_["Qwen3-32B-W8A8"] = 40.0 + 5.0;// 45.0GB
}

void InstanceMgr::init_model_resource_coefficients() {
  gpu_hw_spec_.hbm_per_gpu_gb = FLAGS_gpu_hbm_per_gpu_gb;
  gpu_hw_spec_.compute_sm_per_gpu = FLAGS_gpu_compute_sm_per_gpu;

  // Hardcoded resource coefficients per model
  model_resource_models_["Qwen3-8B"] = std::make_unique<LinearResourceModel>(
      /*hbm_a=*/0.001, /*hbm_b=*/20.0,
      /*compute_a=*/0.0005, /*compute_b=*/0.0);
  model_resource_models_["Qwen2-7B"] = std::make_unique<LinearResourceModel>(
      /*hbm_a=*/0.001, /*hbm_b=*/20.0,
      /*compute_a=*/0.0005, /*compute_b=*/0.0);

  LOG(INFO) << "Initialized model resource models for "
            << model_resource_models_.size() << " models, "
            << "GPU HBM=" << gpu_hw_spec_.hbm_per_gpu_gb << "GB, "
            << "GPU compute SM=" << gpu_hw_spec_.compute_sm_per_gpu;
}

// TODO: support dynamic instance memory specs, rather than hardcoded.
double InstanceMgr::get_model_memory_size(const std::string& model_id) {
    if (model_memory_specs_.count(model_id)) {
        return model_memory_specs_[model_id];
    }
    LOG(WARNING) << "Unknown model ID for memory spec: " << model_id << ", using default 20GB";
    return 20.0; 
}

bool InstanceMgr::is_model_waking_up(const std::string& model_id) {
  auto model_mgr = get_model_instance_mgr(model_id);
  return model_mgr->is_model_waking_up();
}

std::vector<std::string> InstanceMgr::get_awake_instances(const std::string& model_id) {
  auto model_mgr = get_model_instance_mgr(model_id);
  return model_mgr->get_awake_instances();
}

void InstanceMgr::update_model_heat(const std::string& model_id,
                                    int64_t token_count) {
  auto model_mgr = get_model_instance_mgr(model_id);
  model_mgr->update_model_heat(token_count);
}

int32_t InstanceMgr::get_wakeup_count(const std::string& model_id) {
  auto model_mgr = get_model_instance_mgr(model_id);
  return model_mgr->get_wakeup_count();
}


bool InstanceMgr::wait_for_model_drain(const std::string& instance_name,
                                        const std::string& model_id) {
  std::unique_lock<std::mutex> lock(request_metrics_mutex_);
  auto instance_it = request_metrics_.find(instance_name);
  if (instance_it == request_metrics_.end()) {
    LOG(ERROR) << "wait_for_model_drain: no request metrics for " << instance_name;
    return false;
  }
  auto model_it = instance_it->second.model_metrics.find(model_id);
  if (model_it == instance_it->second.model_metrics.end()) {
    LOG(ERROR) << "wait_for_model_drain: no model metrics for " << model_id
               << " on " << instance_name;
    return false;
  }
  auto& metrics = model_it->second;
  metrics.cv_idle.wait(lock, [&metrics]() {
    return metrics.prefill_request_num == 0 && metrics.decode_request_num == 0;
  });
  return true;
}

void InstanceMgr::dynamic_part_auto_scaling() {
  std::unique_lock<std::mutex> alloc_lock(allocation_mutex_);

  int32_t total_gpus = total_available_gpus_.load();
  // Budget: total GPUs minus steady pool reservation
  int32_t budget = std::max(0, total_gpus - steady_needed_gpus());

  // 1. Compute per-model GPU targets (elastic pool models only)
  struct ScalingTarget {
    std::string model_id;
    int64_t heat;
    int32_t gpu_target;
    int32_t gpu_allocated;
    std::shared_ptr<ModelInstanceMgr> mgr;
  };
  std::vector<ScalingTarget> targets;

  {
    std::shared_lock<std::shared_mutex> mgr_lock(model_instance_mgr_mutex_);
    for (const auto& [id, mgr] : model_instance_mgrs_) {
      // Only include elastic pool models
      auto pool_it = model_pool_assignments_.find(id);
      if (pool_it == model_pool_assignments_.end() ||
          pool_it->second != PoolType::ELASTIC) {
        continue;
      }

      ScalingTarget t;
      t.model_id = id;
      t.mgr = mgr;
      t.heat = mgr->get_model_heat();
      auto it = model_resource_models_.find(id);
      t.gpu_target = (t.heat == 0) ? 0
          : (it != model_resource_models_.end())
              ? it->second->compute_gpu_target(t.heat, gpu_hw_spec_)
              : 1;
      t.gpu_allocated = 0;
      targets.push_back(std::move(t));
    }
  }

  // 2. Budget constraint with proportional scaling
  int32_t demand = 0;
  for (auto& t : targets) demand += t.gpu_target;

  if (demand <= budget) {
    for (auto& t : targets) t.gpu_allocated = t.gpu_target;
  } else if (demand > 0) {
    double ratio = static_cast<double>(budget) / demand;
    int32_t sum = 0;
    for (auto& t : targets) {
      t.gpu_allocated = (t.gpu_target > 0)
          ? std::max(1, static_cast<int32_t>(std::floor(t.gpu_target * ratio)))
          : 0;
      sum += t.gpu_allocated;
    }
    // Distribute remainder GPUs using Largest Remainder Method
    // (Hamilton apportionment) to avoid wasting budget.
    int32_t remainder = budget - sum;
    if (remainder > 0) {
      struct RemainderCandidate {
        size_t index;
        double fractional;
        int64_t heat;
      };
      std::vector<RemainderCandidate> candidates;
      for (size_t i = 0; i < targets.size(); ++i) {
        if (targets[i].gpu_target == 0) continue;
        double raw = targets[i].gpu_target * ratio;
        int32_t floored = static_cast<int32_t>(std::floor(raw));
        // Skip models bumped by max(1, ...) guarantee — their fractional
        // part is artificial and should not compete for remainder GPUs.
        if (floored < 1 && targets[i].gpu_allocated == 1) continue;
        double frac = raw - targets[i].gpu_allocated;
        candidates.push_back({i, frac, targets[i].heat});
      }
      // Sort: largest fractional first; tiebreak by lower heat (colder first)
      std::sort(candidates.begin(), candidates.end(),
                [](const RemainderCandidate& a, const RemainderCandidate& b) {
                  if (a.fractional != b.fractional)
                    return a.fractional > b.fractional;
                  return a.heat < b.heat;
                });
      for (int32_t r = 0;
           r < remainder && r < static_cast<int32_t>(candidates.size()); ++r) {
        targets[candidates[r].index].gpu_allocated++;
        sum++;
      }
    }

    // Trim overshoot from lowest-heat models (safety net)
    std::sort(targets.begin(), targets.end(),
              [](const ScalingTarget& a, const ScalingTarget& b) {
                return a.heat < b.heat;
              });
    while (sum > budget) {
      bool trimmed = false;
      for (auto& t : targets) {
        if (t.gpu_allocated > 1) {
          t.gpu_allocated--;
          sum--;
          trimmed = true;
          if (sum <= budget) break;
        }
      }
      if (!trimmed) break;
    }
  }

  for (auto& t : targets) {
    LOG(INFO) << "Scaling plan: model=" << t.model_id
              << " pool=ELASTIC"
              << " heat=" << t.heat
              << " gpu_target=" << t.gpu_target
              << " gpu_allocated=" << t.gpu_allocated
              << " budget=" << budget;
  }

  // 3. Overlapped scale-up/scale-down
  //
  // Phase A: Build scale-down candidates and scale-up needs (round-robin)
  struct ScaleUpNeed {
    size_t target_idx;
    std::string model_id;
    uint64_t model_size;
    bool matched = false;
  };
  struct ScaleDownCandidate {
    std::string instance_name;
    size_t target_idx;
    std::string model_id;
    bool matched = false;
  };

  std::vector<ScaleDownCandidate> scale_down_candidates;
  for (size_t i = 0; i < targets.size(); ++i) {
    int32_t current_wakeup = targets[i].mgr->get_wakeup_count();
    int32_t excess = current_wakeup - targets[i].gpu_allocated;
    if (excess <= 0) continue;
    auto unlocked = targets[i].mgr->get_unlocked_instances();
    for (int32_t j = 0; j < excess &&
         j < static_cast<int32_t>(unlocked.size()); ++j) {
      scale_down_candidates.push_back(
          {unlocked[j], i, targets[i].model_id, false});
    }
  }

  // Build scale-up needs in round-robin order for fairness
  // E.g., A needs +5, B needs +4, C needs +3 → A,B,C,A,B,C,A,B,C,A,B,A
  struct ModelDeficit {
    size_t target_idx;
    int32_t remaining;
    std::string model_id;
    uint64_t model_size;
  };
  std::vector<ModelDeficit> model_deficits;
  for (size_t i = 0; i < targets.size(); ++i) {
    int32_t current_alloc = targets[i].mgr->get_allocation_count();
    int32_t deficit = targets[i].gpu_allocated - current_alloc;
    if (deficit > 0) {
      model_deficits.push_back({i, deficit, targets[i].model_id,
                                get_model_size_bytes(targets[i].model_id)});
    }
  }
  std::vector<ScaleUpNeed> scale_up_needs;
  while (!model_deficits.empty()) {
    for (auto it = model_deficits.begin(); it != model_deficits.end(); ) {
      scale_up_needs.push_back(
          {it->target_idx, it->model_id, it->model_size, false});
      if (--it->remaining <= 0) {
        it = model_deficits.erase(it);
      } else {
        ++it;
      }
    }
  }

  // Phase B: Greedy matching — pair scale-up needs with scale-down candidates
  // Best fit: smallest free_bytes >= model_size (can hold both models at once)
  struct OverlapMatch {
    size_t up_idx;
    size_t down_idx;
  };
  std::vector<OverlapMatch> overlapped_matches;

  for (size_t u = 0; u < scale_up_needs.size(); ++u) {
    auto& up = scale_up_needs[u];
    int best_idx = -1;
    uint64_t best_free = UINT64_MAX;
    for (size_t k = 0; k < scale_down_candidates.size(); ++k) {
      if (scale_down_candidates[k].matched) continue;
      uint64_t free_bytes =
          get_instance_free_bytes(scale_down_candidates[k].instance_name);
      if (free_bytes >= up.model_size && free_bytes < best_free) {
        best_free = free_bytes;
        best_idx = static_cast<int>(k);
      }
    }
    if (best_idx >= 0) {
      overlapped_matches.push_back({u, static_cast<size_t>(best_idx)});
      scale_down_candidates[best_idx].matched = true;
      scale_up_needs[u].matched = true;
    }
  }

  // Phase C: Execute overlapped matches (wake new model while draining old)
  for (auto& m : overlapped_matches) {
    auto& up = scale_up_needs[m.up_idx];
    auto& down = scale_down_candidates[m.down_idx];

    LOG(INFO) << "Overlapped scale: wake " << up.model_id
              << " while draining " << down.model_id
              << " on " << down.instance_name;

    // Mark old model as DRAINING (stops routing new requests to it)
    targets[down.target_idx].mgr->set_model_state(
        down.instance_name, ModelState::DRAINING);

    // Allocate + wake new model on same instance (synchronous)
    targets[up.target_idx].mgr->set_model_state(
        down.instance_name, ModelState::ALLOCATED);
    deduct_free_pages(down.instance_name,
                      get_model_size_bytes(up.model_id));
    send_model_wakeup(down.instance_name, up.model_id, true);

    // Async: drain + sleep old model
    // Instance stays in elastic_occupied_instances_ (new model is using it)
    std::string inst = down.instance_name;
    std::string old_model = down.model_id;
    std::thread([this, inst, old_model]() {
      if (!wait_for_model_drain(inst, old_model)) return;
      auto mgr = get_model_instance_mgr(old_model);
      if (mgr && !mgr->can_sleep(inst)) return;
      send_model_sleep(inst, old_model);
    }).detach();
  }

  // Phase D: Execute remaining scale-ups (on free instances)
  for (auto& t : targets) {
    int32_t current_alloc = t.mgr->get_allocation_count();
    int32_t deficit = t.gpu_allocated - current_alloc;
    if (deficit <= 0) continue;

    uint64_t model_size = get_model_size_bytes(t.model_id);

    std::shared_lock<std::shared_mutex> inst_lock(inst_mutex_);
    for (const auto& [inst_name, _] : instances_) {
      if (deficit <= 0) break;
      if (t.mgr->get_model_state(inst_name) != ModelState::SLEEP) continue;
      if (is_steady_pool_instance(inst_name)) continue;
      if (elastic_occupied_instances_.count(inst_name)) continue;
      if (!has_valid_xtensor_info(inst_name)) {
        LOG(WARNING) << "dynamic_part_auto_scaling: skip " << inst_name
                     << " for model " << t.model_id
                     << " (no valid xtensor info)";
        continue;
      }
      if (get_instance_free_bytes(inst_name) < model_size) {
        LOG(INFO) << "dynamic_part_auto_scaling: skip " << inst_name
                  << " for model " << t.model_id << " (not enough space: free="
                  << get_instance_free_bytes(inst_name)
                  << " need=" << model_size << ")";
        continue;
      }

      t.mgr->set_model_state(inst_name, ModelState::ALLOCATED);
      deduct_free_pages(inst_name, model_size);
      elastic_occupied_instances_.insert(inst_name);
      inst_lock.unlock();

      send_model_wakeup(inst_name, t.model_id, true);  // blocking
      deficit--;

      if (deficit > 0) inst_lock.lock();
    }
  }

  // Phase E: Execute remaining scale-downs (async drain + sleep)
  for (auto& down : scale_down_candidates) {
    if (down.matched) continue;
    targets[down.target_idx].mgr->set_model_state(
        down.instance_name, ModelState::DRAINING);
    std::string inst = down.instance_name;
    std::string model = down.model_id;
    std::thread([this, inst, model]() {
      if (!wait_for_model_drain(inst, model)) return;
      auto mgr = get_model_instance_mgr(model);
      if (mgr && !mgr->can_sleep(inst)) return;
      send_model_sleep(inst, model);
      std::lock_guard<std::mutex> lock(allocation_mutex_);
      elastic_occupied_instances_.erase(inst);
    }).detach();
  }

  // Cleanup: if heat → 0, remove from pool
  for (auto& t : targets) {
    if (t.gpu_allocated == 0) {
      model_pool_assignments_[t.model_id] = PoolType::NONE;
      LOG(INFO) << "Model " << t.model_id
                << " heat=0, removed from ELASTIC pool";
    }
  }
}

std::shared_ptr<ModelInstanceMgr> InstanceMgr::get_model_instance_mgr(const std::string& model_id) {
  std::shared_lock<std::shared_mutex> lock(model_instance_mgr_mutex_);
  auto it = model_instance_mgrs_.find(model_id);
  if (it != model_instance_mgrs_.end()) {
    return it->second;
  }
  LOG(ERROR) << "Model instance manager not found for model " << model_id;
  return nullptr;
}

bool InstanceMgr::has_valid_xtensor_info(const std::string& instance_name) {
  std::lock_guard<std::mutex> lock(xtensor_info_mutex_);
  auto it = instance_xtensor_infos_.find(instance_name);
  return it != instance_xtensor_infos_.end() && it->second.is_valid;
}

uint64_t InstanceMgr::get_model_size_bytes(const std::string& model_id) {
  // Priority 1: Get from any instance's xtensor info
  {
    std::lock_guard<std::mutex> lock(xtensor_info_mutex_);
    for (const auto& [inst_name, info] : instance_xtensor_infos_) {
      if (!info.is_valid) continue;
      uint64_t size = info.get_model_size_bytes(model_id);
      if (size > 0) return size;
    }
  }

  // Priority 2: Fallback to model_memory_specs_
  if (model_memory_specs_.count(model_id)) {
    return static_cast<uint64_t>(model_memory_specs_[model_id] * 1024 * 1024 * 1024);
  }

  // Default fallback: 20GB
  LOG(WARNING) << "Unknown model size for " << model_id << ", using default 20GB";
  return 20ULL * 1024 * 1024 * 1024;
}

uint64_t InstanceMgr::get_instance_free_bytes(const std::string& instance_name) {
  std::lock_guard<std::mutex> lock(xtensor_info_mutex_);
  auto it = instance_xtensor_infos_.find(instance_name);
  if (it == instance_xtensor_infos_.end() || !it->second.is_valid) {
    return 0;  // No valid info, cannot allocate
  }
  return it->second.get_min_free_bytes();
}

bool InstanceMgr::has_enough_space_for_model(const std::string& instance_name,
                                              const std::string& model_id) {
  uint64_t free_bytes = get_instance_free_bytes(instance_name);
  uint64_t model_size = get_model_size_bytes(model_id);
  return free_bytes >= model_size;
}

void InstanceMgr::deduct_free_pages(const std::string& instance_name, uint64_t bytes) {
  std::lock_guard<std::mutex> lock(xtensor_info_mutex_);
  auto it = instance_xtensor_infos_.find(instance_name);
  if (it == instance_xtensor_infos_.end() || !it->second.is_valid) return;

  uint64_t pages_to_deduct = (bytes + kXTensorPageSizeBytes - 1) / kXTensorPageSizeBytes;
  for (auto& pages : it->second.worker_free_phy_pages) {
    pages = (pages > pages_to_deduct) ? (pages - pages_to_deduct) : 0;
  }
}

void InstanceMgr::update_xtensor_info(
    const std::string& instance_name,
    const proto::XTensorHeartbeatInfo& xtensor_info) {
  std::lock_guard<std::mutex> lock(xtensor_info_mutex_);

  InstanceXTensorInfo info;
  info.is_valid = true;  // Mark as valid heartbeat data

  // Copy worker_free_phy_pages
  for (auto pages : xtensor_info.worker_free_phy_pages()) {
    info.worker_free_phy_pages.push_back(pages);
  }

  // Copy model_weight_segments
  for (const auto& [model_id, segment_list] : xtensor_info.model_weight_segments()) {
    std::vector<WeightSegment> segments;
    for (const auto& seg : segment_list.segments()) {
      segments.push_back({seg.offset(), seg.size()});
    }
    info.model_weight_segments[model_id] = std::move(segments);
  }

  // Copy device addresses for D2D transfer
  // Prefer heartbeat-reported addresses; preserve registration-time addresses if heartbeat doesn't include them
  if (xtensor_info.device_addrs_size() > 0) {
    for (const auto& addr : xtensor_info.device_addrs()) {
      info.device_addrs.push_back(addr);
    }
  } else {
    auto it = instance_xtensor_infos_.find(instance_name);
    if (it != instance_xtensor_infos_.end()) {
      info.device_addrs = it->second.device_addrs;
    }
  }

  // Preserve p2p_addrs from registration (not reported via heartbeat)
  {
    auto it = instance_xtensor_infos_.find(instance_name);
    if (it != instance_xtensor_infos_.end()) {
      info.p2p_addrs = it->second.p2p_addrs;
    }
  }

  instance_xtensor_infos_[instance_name] = std::move(info);
}

std::optional<InstanceXTensorInfo> InstanceMgr::get_instance_xtensor_info(
    const std::string& instance_name) {
  std::lock_guard<std::mutex> lock(xtensor_info_mutex_);
  auto it = instance_xtensor_infos_.find(instance_name);
  if (it != instance_xtensor_infos_.end()) {
    return it->second;
  }
  return std::nullopt;
}

std::optional<D2DWakeupInfo> InstanceMgr::find_d2d_source(
    const std::string& model_id,
    const std::string& target_instance_name) {
  // Find instances that have the model in WAKEUP state
  auto model_mgr = get_model_instance_mgr(model_id);
  if (!model_mgr) {
    LOG(WARNING) << "Model manager not found for " << model_id;
    return std::nullopt;
  }

  // Atomically get awake instances and lock all of them
  auto locked_instances = model_mgr->get_awake_instances_and_lock();
  if (locked_instances.empty()) {
    LOG(INFO) << "No awake instances found for model " << model_id
              << ", will use H2D wakeup";
    return std::nullopt;
  }

  // Track which instance we select as source (to keep it locked)
  std::string selected_source;

  // Collect all eligible D2D source candidates, then pick the most loaded one.
  struct D2DCandidate {
    std::string instance_name;
    std::vector<std::string> p2p_addrs;
    std::vector<WeightSegment> weight_segments;
    int64_t epdt;  // estimated prefill done time (higher = more loaded)
  };
  std::vector<D2DCandidate> candidates;

  // Find all suitable source instances (not the target itself)
  for (const auto& source_instance_name : locked_instances) {
    if (source_instance_name == target_instance_name) {
      continue;
    }

    // Check if we have XTensor info for this source instance
    std::optional<InstanceXTensorInfo> xtensor_info;
    {
      std::lock_guard<std::mutex> lock(xtensor_info_mutex_);
      auto it = instance_xtensor_infos_.find(source_instance_name);
      if (it != instance_xtensor_infos_.end() && it->second.is_valid) {
        xtensor_info = it->second;
      }
    }

    if (!xtensor_info.has_value()) {
      continue;
    }

    // Check if the source has weight segments for this model
    auto seg_it = xtensor_info->model_weight_segments.find(model_id);
    if (seg_it == xtensor_info->model_weight_segments.end() ||
        seg_it->second.empty()) {
      continue;
    }

    // Get P2P addresses for D2D weight transfer
    std::vector<std::string> p2p_addrs;
    if (!xtensor_info->p2p_addrs.empty()) {
      p2p_addrs = xtensor_info->p2p_addrs;
    } else {
      std::shared_lock<std::shared_mutex> lock(inst_mutex_);
      auto meta_it = instances_.find(source_instance_name);
      if (meta_it != instances_.end()) {
        p2p_addrs = meta_it->second.p2p_addrs;
      }
    }

    if (p2p_addrs.empty()) {
      continue;
    }

    // Eligible candidate — record with EPDT for load ranking
    int64_t epdt = get_estimated_prefill_done_time(source_instance_name);
    candidates.push_back({source_instance_name, std::move(p2p_addrs),
                          seg_it->second, epdt});
  }

  if (candidates.empty()) {
    model_mgr->release_d2d_locks(locked_instances);
    LOG(INFO) << "No suitable D2D source found for model " << model_id
              << ", will use H2D wakeup";
    return std::nullopt;
  }

  // Prefer the most heavily-loaded instance (highest EPDT) as D2D source.
  // This avoids choosing lightly-loaded instances that are more likely to be
  // drained/evicted.
  auto& best = *std::max_element(
      candidates.begin(), candidates.end(),
      [](const D2DCandidate& a, const D2DCandidate& b) {
        return a.epdt < b.epdt;
      });

  // Build D2DWakeupInfo from best candidate
  D2DWakeupInfo d2d_info;
  d2d_info.source_instance_name = best.instance_name;
  d2d_info.remote_addrs = best.p2p_addrs;
  for (size_t i = 0; i < best.p2p_addrs.size(); ++i) {
    d2d_info.src_weight_segments.push_back(best.weight_segments);
  }
  selected_source = best.instance_name;

  LOG(INFO) << "Found D2D source instance " << selected_source
            << " for model " << model_id
            << " with " << d2d_info.remote_addrs.size() << " remote addrs"
            << " and " << best.weight_segments.size() << " weight segments"
            << " (EPDT=" << best.epdt << ", candidates=" << candidates.size() << ")";

  // Unlock all instances except the selected source
  std::vector<std::string> instances_to_unlock;
  for (const auto& inst : locked_instances) {
    if (inst != selected_source) {
      instances_to_unlock.push_back(inst);
    }
  }
  model_mgr->release_d2d_locks(instances_to_unlock);

  return d2d_info;
}

// --- Dual-pool scheduling implementation ---

PoolType InstanceMgr::get_model_pool(const std::string& model_id) {
  std::lock_guard<std::mutex> lock(allocation_mutex_);
  auto it = model_pool_assignments_.find(model_id);
  if (it != model_pool_assignments_.end()) {
    return it->second;
  }
  return PoolType::NONE;
}

bool InstanceMgr::is_steady_pool_instance(const std::string& instance_name) {
  for (const auto& bin : steady_bins_) {
    if (bin.instance_name == instance_name) return true;
  }
  return false;
}

int32_t InstanceMgr::steady_needed_gpus() {
  return static_cast<int32_t>(steady_bins_.size()) * kTensorParallelSize
      + pending_steady_gpus_;
}

ResourceNeeds InstanceMgr::get_model_resource_needs(const std::string& model_id) {
  auto model_mgr = get_model_instance_mgr(model_id);
  int64_t heat = model_mgr ? model_mgr->get_model_heat() : 0;

  auto it = model_resource_models_.find(model_id);
  if (it != model_resource_models_.end()) {
    return it->second->compute_resource_needs(heat);
  }
  // Default: use hbm_b=20GB, compute_b=0 for unknown models
  return {20.0, 0.0};
}

void InstanceMgr::assign_model_to_pool(const std::string& model_id) {
  std::unique_lock<std::mutex> alloc_lock(allocation_mutex_);

  // Already assigned?
  if (model_pool_assignments_.count(model_id) &&
      model_pool_assignments_[model_id] != PoolType::NONE) {
    return;
  }

  auto model_mgr = get_model_instance_mgr(model_id);
  if (!model_mgr) {
    LOG(ERROR) << "assign_model_to_pool: no model mgr for " << model_id;
    return;
  }

  int64_t heat = model_mgr->get_model_heat();
  auto res_it = model_resource_models_.find(model_id);
  int32_t gpu_target = (heat == 0) ? 1
      : (res_it != model_resource_models_.end())
          ? res_it->second->compute_gpu_target(heat, gpu_hw_spec_)
          : 1;

  LOG(INFO) << "assign_model_to_pool: model=" << model_id
            << " heat=" << heat << " gpu_target=" << gpu_target;

  if (gpu_target <= 1) {
    // Steady pool
    ResourceNeeds needs = (res_it != model_resource_models_.end())
        ? res_it->second->compute_resource_needs(heat)
        : ResourceNeeds{20.0, 0.0};

    std::string instance = find_or_create_steady_bin(model_id, needs);
    if (instance.empty()) {
      LOG(WARNING) << "assign_model_to_pool: no instance for steady pool, "
                   << "falling back to elastic for " << model_id;
      model_pool_assignments_[model_id] = PoolType::ELASTIC;
      return;
    }

    model_pool_assignments_[model_id] = PoolType::STEADY;

    uint64_t model_size = get_model_size_bytes(model_id);
    model_mgr->set_model_state(instance, ModelState::ALLOCATED);
    deduct_free_pages(instance, model_size);

    alloc_lock.unlock();
    send_model_wakeup(instance, model_id, true);

    LOG(INFO) << "Assigned model " << model_id << " to STEADY pool on " << instance;
  } else {
    // Elastic pool
    model_pool_assignments_[model_id] = PoolType::ELASTIC;
    LOG(INFO) << "Assigned model " << model_id << " to ELASTIC pool (gpu_target="
              << gpu_target << ")";
  }
}

std::string InstanceMgr::find_or_create_steady_bin(
    const std::string& model_id, const ResourceNeeds& needs,
    bool allow_reclaim) {

  // Phase 1: First Fit in existing bins
  for (auto& bin : steady_bins_) {
    if (bin.remaining_hbm_gb >= needs.hbm_gb &&
        bin.remaining_compute_sm >= needs.compute_sm) {
      bin.remaining_hbm_gb -= needs.hbm_gb;
      bin.remaining_compute_sm -= needs.compute_sm;
      bin.models.insert(model_id);
      LOG(INFO) << "Placed model " << model_id << " in existing steady bin "
                << bin.instance_name << " (remaining hbm=" << bin.remaining_hbm_gb
                << "GB, compute=" << bin.remaining_compute_sm << ")";
      return bin.instance_name;
    }
  }

  // Phase 2: Open new bin from idle instances
  {
    std::shared_lock<std::shared_mutex> inst_lock(inst_mutex_);
    for (const auto& [inst_name, _] : instances_) {
      if (is_steady_pool_instance(inst_name)) {
        continue;
      }
      if (elastic_occupied_instances_.count(inst_name)) {
        continue;
      }
      if (!has_valid_xtensor_info(inst_name)) {
        continue;
      }

      // Check that no model is loaded on this instance
      bool is_idle = true;
      {
        std::shared_lock<std::shared_mutex> mgr_lock(model_instance_mgr_mutex_);
        for (const auto& [mid, mgr] : model_instance_mgrs_) {
          if (mgr->get_model_state(inst_name) != ModelState::SLEEP) {
            is_idle = false;
            break;
          }
        }
      }
      if (!is_idle) continue;

      SteadyBin new_bin;
      new_bin.instance_name = inst_name;
      new_bin.remaining_hbm_gb = gpu_hw_spec_.hbm_per_gpu_gb - needs.hbm_gb;
      new_bin.remaining_compute_sm = gpu_hw_spec_.compute_sm_per_gpu - needs.compute_sm;
      new_bin.models.insert(model_id);
      steady_bins_.push_back(std::move(new_bin));

      LOG(INFO) << "Created new steady bin on " << inst_name << " for model "
                << model_id;
      return inst_name;
    }
  }

  // Phase 3: No idle instance — reclaim from elastic pool (BLOCKING)
  if (!allow_reclaim) {
    return "";
  }
  pending_steady_gpus_ += kTensorParallelSize;

  // Release allocation_mutex_ since dynamic_part_auto_scaling also acquires it
  // Note: caller must handle this unlock/relock pattern
  allocation_mutex_.unlock();
  dynamic_part_auto_scaling();
  allocation_mutex_.lock();

  pending_steady_gpus_ -= kTensorParallelSize;

  // Retry: find the freed instance
  {
    std::shared_lock<std::shared_mutex> inst_lock(inst_mutex_);
    for (const auto& [inst_name, _] : instances_) {
      if (is_steady_pool_instance(inst_name)) continue;
      if (elastic_occupied_instances_.count(inst_name)) continue;
      if (!has_valid_xtensor_info(inst_name)) continue;

      bool is_idle = true;
      {
        std::shared_lock<std::shared_mutex> mgr_lock(model_instance_mgr_mutex_);
        for (const auto& [mid, mgr] : model_instance_mgrs_) {
          if (mgr->get_model_state(inst_name) != ModelState::SLEEP) {
            is_idle = false;
            break;
          }
        }
      }
      if (!is_idle) continue;

      SteadyBin new_bin;
      new_bin.instance_name = inst_name;
      new_bin.remaining_hbm_gb = gpu_hw_spec_.hbm_per_gpu_gb - needs.hbm_gb;
      new_bin.remaining_compute_sm = gpu_hw_spec_.compute_sm_per_gpu - needs.compute_sm;
      new_bin.models.insert(model_id);
      steady_bins_.push_back(std::move(new_bin));

      LOG(INFO) << "Reclaimed instance " << inst_name << " for steady bin, model "
                << model_id;
      return inst_name;
    }
  }

  return "";
}

void InstanceMgr::remove_model_from_steady_bin(const std::string& model_id) {
  // Must be called with allocation_mutex_ held
  for (auto it = steady_bins_.begin(); it != steady_bins_.end(); ++it) {
    if (it->models.count(model_id)) {
      // Reclaim resources
      ResourceNeeds needs = get_model_resource_needs(model_id);
      it->remaining_hbm_gb += needs.hbm_gb;
      it->remaining_compute_sm += needs.compute_sm;
      it->models.erase(model_id);

      LOG(INFO) << "Removed model " << model_id << " from steady bin "
                << it->instance_name;

      // If bin is now empty, remove it
      if (it->models.empty()) {
        LOG(INFO) << "Steady bin " << it->instance_name << " is now empty, releasing";
        steady_bins_.erase(it);
      }
      return;
    }
  }
}

bool InstanceMgr::steady_part_check_upgrading(const std::string& model_id) {
  std::unique_lock<std::mutex> alloc_lock(allocation_mutex_);

  auto pool_it = model_pool_assignments_.find(model_id);
  if (pool_it == model_pool_assignments_.end() ||
      pool_it->second != PoolType::STEADY) {
    return false;
  }

  auto model_mgr = get_model_instance_mgr(model_id);
  if (!model_mgr) return false;

  int64_t heat = model_mgr->get_model_heat();
  auto res_it = model_resource_models_.find(model_id);
  int32_t gpu_target = (res_it != model_resource_models_.end())
      ? res_it->second->compute_gpu_target(heat, gpu_hw_spec_)
      : 1;

  if (gpu_target <= 1) {
    return false;  // still fits in steady pool
  }

  // Find the steady instance hosting this model
  std::string steady_instance;
  for (const auto& bin : steady_bins_) {
    if (bin.models.count(model_id)) {
      steady_instance = bin.instance_name;
      break;
    }
  }

  // Don't upgrade while the model wakeup is still in progress (ALLOCATED state).
  // The drain thread would fail (ALLOCATED→DRAINING is invalid), and
  // dynamic_part_auto_scaling would see the in-progress allocation as already
  // satisfying the budget, resulting in no new instances being woken.
  if (!steady_instance.empty() &&
      model_mgr->get_model_state(steady_instance) != ModelState::WAKEUP) {
    return false;
  }

  // Upgrade to elastic
  pool_it->second = PoolType::ELASTIC;

  remove_model_from_steady_bin(model_id);

  alloc_lock.unlock();

  LOG(INFO) << "Upgrading model " << model_id << " from STEADY to ELASTIC pool "
            << "(gpu_target=" << gpu_target << ")";

  // Async: drain + sleep the model on old steady instance
  if (!steady_instance.empty()) {
    std::string inst = steady_instance;
    std::string mid = model_id;
    std::thread([this, inst, mid]() {
      auto mgr = get_model_instance_mgr(mid);
      if (mgr) mgr->set_model_state(inst, ModelState::DRAINING);
      if (!wait_for_model_drain(inst, mid)) return;
      send_model_sleep(inst, mid);
    }).detach();
  }

  return true;
}

bool InstanceMgr::route_to_steady_instance(std::shared_ptr<Request> request) {
  std::lock_guard<std::mutex> lock(allocation_mutex_);
  for (const auto& bin : steady_bins_) {
    if (bin.models.count(request->model)) {
      request->routing.prefill_name = bin.instance_name;
      request->routing.decode_name = bin.instance_name;
      return true;
    }
  }
  LOG(ERROR) << "route_to_steady_instance: model " << request->model
             << " not found in any steady bin";
  return false;
}

std::vector<std::tuple<std::string, std::string, std::string>>
InstanceMgr::repack_steady_bins() {
  // Must be called with allocation_mutex_ held
  struct PackItem {
    std::string model_id;
    ResourceNeeds needs;
    std::string old_instance;
    double dominant_ratio;
  };

  // Collect all items from current bins
  std::vector<PackItem> items;
  for (const auto& bin : steady_bins_) {
    for (const auto& model_id : bin.models) {
      PackItem item;
      item.model_id = model_id;
      item.needs = get_model_resource_needs(model_id);
      item.old_instance = bin.instance_name;
      item.dominant_ratio = std::max(
          item.needs.hbm_gb / gpu_hw_spec_.hbm_per_gpu_gb,
          item.needs.compute_sm / gpu_hw_spec_.compute_sm_per_gpu);
      items.push_back(std::move(item));
    }
  }

  if (items.empty()) {
    steady_bins_.clear();
    return {};
  }

  // Sort by dominant resource ratio (decreasing) — FFD heuristic
  std::sort(items.begin(), items.end(),
            [](const PackItem& a, const PackItem& b) {
              return a.dominant_ratio > b.dominant_ratio;
            });

  // Collect old bin instance names for reuse
  std::vector<std::string> old_instances;
  for (const auto& bin : steady_bins_) {
    old_instances.push_back(bin.instance_name);
  }

  // Build new bins with FFD
  std::vector<SteadyBin> new_bins;
  size_t next_old_instance = 0;

  for (const auto& item : items) {
    bool placed = false;
    for (auto& bin : new_bins) {
      if (bin.remaining_hbm_gb >= item.needs.hbm_gb &&
          bin.remaining_compute_sm >= item.needs.compute_sm) {
        bin.remaining_hbm_gb -= item.needs.hbm_gb;
        bin.remaining_compute_sm -= item.needs.compute_sm;
        bin.models.insert(item.model_id);
        placed = true;
        break;
      }
    }
    if (!placed) {
      // Open new bin, preferring old steady instances
      std::string inst_name;
      if (next_old_instance < old_instances.size()) {
        inst_name = old_instances[next_old_instance++];
      } else {
        // Need a completely new instance — find idle
        std::shared_lock<std::shared_mutex> inst_lock(inst_mutex_);
        for (const auto& [name, _] : instances_) {
          if (elastic_occupied_instances_.count(name)) continue;
          if (!has_valid_xtensor_info(name)) continue;
          // Check not already used in new_bins
          bool already_used = false;
          for (const auto& nb : new_bins) {
            if (nb.instance_name == name) { already_used = true; break; }
          }
          if (already_used) continue;
          inst_name = name;
          break;
        }
      }

      if (inst_name.empty()) {
        LOG(ERROR) << "repack_steady_bins: no instance available for model "
                   << item.model_id;
        continue;
      }

      SteadyBin new_bin;
      new_bin.instance_name = inst_name;
      new_bin.remaining_hbm_gb = gpu_hw_spec_.hbm_per_gpu_gb - item.needs.hbm_gb;
      new_bin.remaining_compute_sm = gpu_hw_spec_.compute_sm_per_gpu - item.needs.compute_sm;
      new_bin.models.insert(item.model_id);
      new_bins.push_back(std::move(new_bin));
    }
  }

  // Compute moves
  std::vector<std::tuple<std::string, std::string, std::string>> moves;
  for (const auto& item : items) {
    std::string new_instance;
    for (const auto& bin : new_bins) {
      if (bin.models.count(item.model_id)) {
        new_instance = bin.instance_name;
        break;
      }
    }
    if (!new_instance.empty() && new_instance != item.old_instance) {
      moves.emplace_back(item.model_id, item.old_instance, new_instance);
    }
  }

  steady_bins_ = std::move(new_bins);

  LOG(INFO) << "Steady pool repack: " << items.size() << " models, "
            << steady_bins_.size() << " bins, " << moves.size() << " moves";

  return moves;
}

void InstanceMgr::steady_part_auto_repacking() {
  std::unique_lock<std::mutex> alloc_lock(allocation_mutex_);

  // Phase 1: collect models with 0 heat → schedule sleep
  std::vector<std::pair<std::string, std::string>> models_to_remove;  // (model_id, instance)
  for (const auto& bin : steady_bins_) {
    for (const auto& model_id : bin.models) {
      auto model_mgr = get_model_instance_mgr(model_id);
      if (model_mgr && model_mgr->get_model_heat() == 0) {
        models_to_remove.emplace_back(model_id, bin.instance_name);
      }
    }
  }

  for (const auto& [model_id, instance] : models_to_remove) {
    remove_model_from_steady_bin(model_id);
    model_pool_assignments_.erase(model_id);
    LOG(INFO) << "steady_part_auto_repacking: sleeping model " << model_id
              << " on " << instance << " (heat=0)";
    std::string inst = instance;
    std::string mid = model_id;
    std::thread([this, inst, mid]() {
      send_model_sleep(inst, mid);
    }).detach();
  }

  // Phase 2: FFD repack of remaining models
  auto moves = repack_steady_bins();

  alloc_lock.unlock();

  // Phase 3: Execute moves (wake on new, drain+sleep on old)
  for (const auto& [model_id, old_inst, new_inst] : moves) {
    LOG(INFO) << "steady_part_auto_repacking: moving model " << model_id
              << " from " << old_inst << " to " << new_inst;
    send_model_wakeup(new_inst, model_id, false);
    auto mgr = get_model_instance_mgr(model_id);
    if (mgr) mgr->set_model_state(old_inst, ModelState::DRAINING);
    wait_for_model_drain(old_inst, model_id);
    send_model_sleep(old_inst, model_id);
  }
}

void InstanceMgr::elastic_to_steady_demotion() {
  static constexpr int kDemotionThresholdSeconds = 30;
  auto now = std::chrono::steady_clock::now();

  // Phase 1: Scan elastic models and update low-demand tracking
  struct DemotionCandidate {
    std::string model_id;
    std::shared_ptr<ModelInstanceMgr> mgr;
  };
  std::vector<DemotionCandidate> candidates;

  {
    std::unique_lock<std::mutex> alloc_lock(allocation_mutex_);

    std::vector<std::string> elastic_models;
    for (const auto& [id, pool] : model_pool_assignments_) {
      if (pool == PoolType::ELASTIC) {
        elastic_models.push_back(id);
      }
    }

    for (const auto& model_id : elastic_models) {
      auto model_mgr = get_model_instance_mgr(model_id);
      if (!model_mgr) continue;

      int64_t heat = model_mgr->get_model_heat();
      auto res_it = model_resource_models_.find(model_id);
      int32_t gpu_target = (heat == 0) ? 0
          : (res_it != model_resource_models_.end())
              ? res_it->second->compute_gpu_target(heat, gpu_hw_spec_)
              : 1;

      if (gpu_target != 1) {
        // Demand is not "low but nonzero" — reset tracking
        elastic_low_demand_since_.erase(model_id);
        continue;
      }

      // gpu_target == 1: track low demand duration
      auto it = elastic_low_demand_since_.find(model_id);
      if (it == elastic_low_demand_since_.end()) {
        elastic_low_demand_since_[model_id] = now;
        continue;
      }

      auto duration_s = std::chrono::duration_cast<std::chrono::seconds>(
          now - it->second).count();
      if (duration_s < kDemotionThresholdSeconds) {
        continue;
      }

      // Sustained low demand — mark as demotion candidate
      candidates.push_back({model_id, model_mgr});
    }
  }  // release allocation_mutex_

  // Phase 2: Attempt demotion for each candidate
  for (const auto& candidate : candidates) {
    const auto& model_id = candidate.model_id;
    auto model_mgr = candidate.mgr;

    std::unique_lock<std::mutex> alloc_lock(allocation_mutex_);

    // Re-verify: still ELASTIC and gpu_target still <= 1
    auto pool_it = model_pool_assignments_.find(model_id);
    if (pool_it == model_pool_assignments_.end() ||
        pool_it->second != PoolType::ELASTIC) {
      elastic_low_demand_since_.erase(model_id);
      continue;
    }

    int64_t heat = model_mgr->get_model_heat();
    auto res_it = model_resource_models_.find(model_id);
    int32_t gpu_target = (heat == 0) ? 0
        : (res_it != model_resource_models_.end())
            ? res_it->second->compute_gpu_target(heat, gpu_hw_spec_)
            : 1;
    if (gpu_target != 1) {
      elastic_low_demand_since_.erase(model_id);
      continue;
    }

    // Try to find a steady bin (no reclamation allowed)
    ResourceNeeds needs = get_model_resource_needs(model_id);
    std::string steady_instance =
        find_or_create_steady_bin(model_id, needs, /*allow_reclaim=*/false);

    if (steady_instance.empty()) {
      LOG(INFO) << "elastic_to_steady_demotion: no room in steady pool for "
                << model_id << ", skipping this round";
      continue;
    }

    // Collect elastic instances before changing pool
    auto elastic_instances = model_mgr->get_awake_instances();

    // Set model state to ALLOCATED on steady instance, deduct memory
    uint64_t model_size = get_model_size_bytes(model_id);
    model_mgr->set_model_state(steady_instance, ModelState::ALLOCATED);
    deduct_free_pages(steady_instance, model_size);

    // Change pool assignment to STEADY
    pool_it->second = PoolType::STEADY;

    LOG(INFO) << "elastic_to_steady_demotion: demoting model " << model_id
              << " from ELASTIC to STEADY on " << steady_instance
              << " (elastic instances: " << elastic_instances.size() << ")";

    alloc_lock.unlock();

    // Blocking wakeup on steady instance
    send_model_wakeup(steady_instance, model_id, true);

    // Check if wakeup succeeded
    if (model_mgr->get_model_state(steady_instance) != ModelState::WAKEUP) {
      LOG(ERROR) << "elastic_to_steady_demotion: wakeup failed for "
                 << model_id << " on " << steady_instance << ", rolling back";
      std::lock_guard<std::mutex> lock(allocation_mutex_);
      remove_model_from_steady_bin(model_id);
      model_pool_assignments_[model_id] = PoolType::ELASTIC;
      elastic_low_demand_since_.erase(model_id);
      continue;
    }

    // Drain and sleep all elastic instances for this model
    for (const auto& elastic_inst : elastic_instances) {
      if (elastic_inst == steady_instance) continue;
      model_mgr->set_model_state(elastic_inst, ModelState::DRAINING);
      wait_for_model_drain(elastic_inst, model_id);
      send_model_sleep(elastic_inst, model_id);
      {
        std::lock_guard<std::mutex> lock(allocation_mutex_);
        elastic_occupied_instances_.erase(elastic_inst);
      }
    }

    {
      std::lock_guard<std::mutex> lock(allocation_mutex_);
      elastic_low_demand_since_.erase(model_id);
    }

    LOG(INFO) << "elastic_to_steady_demotion: model " << model_id
              << " successfully demoted to STEADY on " << steady_instance;
  }
}

}  // namespace xllm_service
