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
#include <limits>
#include <shared_mutex>

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

InstanceMgr::~InstanceMgr() { exited_ = true; }

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
          LOG(INFO) << "[EPDT] FINISH_PREFILL correction: instance="
                    << request->routing.prefill_name
                    << " expected=" << request->expected_prefill_done_ms
                    << " actual=" << now_ms
                    << " correction=" << correction << "ms";
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

// returns the instance_names of newly allocated(already wakeup or is waking_up) models
std::vector<std::string> InstanceMgr::allocate_instance_for_model(const std::string& model_id,
                                                                  int32_t target_model_count) {

  std::unique_lock<std::mutex> allocation_lock(allocation_mutex_);
  // check for race conditions
  // (multiple entrance in allocate_instance_for_model)
  int model_count_margin = 0;
  {
    std::shared_lock<std::shared_mutex> inst_lock(inst_mutex_);
    auto model_mgr = get_model_instance_mgr(model_id);
    model_count_margin = target_model_count - model_mgr->get_allocation_count();
    if (model_count_margin <= 0) {
      LOG(INFO) << "Model " << model_id << " is already being allocated to target count "
                << target_model_count << ". Give up allocation.";
      return {};
    }
  }

  uint64_t model_size_bytes = get_model_size_bytes(model_id);

  LOG(INFO) << "Allocating instance for model " << model_id
            << " with size " << (model_size_bytes / (1024.0 * 1024 * 1024)) << " GB."
            << " Model count margin: " << model_count_margin;

  std::vector<std::thread> wakeup_threads;
  int newly_allocated_count = 0;
  std::vector<std::string> newly_allocated_instances;

  // First, try instances with enough free space (using xtensor info)
  {
    std::shared_lock<std::shared_mutex> inst_lock(inst_mutex_);
    for (const auto& inst_pair : instances_) {
      const std::string& instance_name = inst_pair.first;
      auto model_mgr = get_model_instance_mgr(model_id);

      if (model_mgr->get_model_state(instance_name) != ModelState::SLEEP) {
        // Model not ready to wake up.
        continue;
      }

      // Skip instances without valid xtensor info
      if (!has_valid_xtensor_info(instance_name)) {
        LOG(INFO) << "Instance " << instance_name << " has no valid xtensor info, skipping";
        continue;
      }

      uint64_t free_bytes = get_instance_free_bytes(instance_name);

      LOG(INFO) << "Instance " << instance_name
                << " free space: " << (free_bytes / (1024.0 * 1024 * 1024)) << " GB.";

      if (free_bytes >= model_size_bytes) {
        model_mgr->set_model_state(instance_name, ModelState::ALLOCATED);

        // Locally deduct free pages to prevent double allocation before next heartbeat
        deduct_free_pages(instance_name, model_size_bytes);

        wakeup_threads.emplace_back([this, instance_name, model_id]() {
          send_model_wakeup(instance_name, model_id, /*memory_increased_in_advance*/ true);
        });
        newly_allocated_count += 1;
        newly_allocated_instances.push_back(instance_name);
        if (newly_allocated_count >= model_count_margin) {
          break;
        }
      }
    }
  }

  // Strategy: Find the instance where we can free up enough space by evicting the *coldest* models.

  std::vector<EvictionPlanInfo> eviction_plans;

  if (newly_allocated_count < model_count_margin) {// At least evict one instance
    std::shared_lock<std::shared_mutex> inst_lock(inst_mutex_); // Iterate instances safely
    for (const auto& inst_pair : instances_) {
      std::string instance_name = inst_pair.first;
      auto model_mgr = get_model_instance_mgr(model_id);

      if (model_mgr->get_model_state(instance_name) != ModelState::SLEEP) {
        continue;
      }

      // Skip instances without valid xtensor info
      if (!has_valid_xtensor_info(instance_name)) {
        LOG(INFO) << "Instance " << instance_name << " has no valid xtensor info, skipping for eviction";
        continue;
      }

      uint64_t free_bytes = get_instance_free_bytes(instance_name);
      // space_needed is how many more bytes we need beyond what's already free
      int64_t space_needed_bytes = static_cast<int64_t>(model_size_bytes) - static_cast<int64_t>(free_bytes);
      double space_needed_gb = space_needed_bytes / (1024.0 * 1024.0 * 1024.0);

      EvictionPlanInfo current_plan = select_eviction_candidates(instance_name, space_needed_gb);
      if (current_plan.instance_name.empty()) {
        continue; // Cannot free enough space on this instance
      }

      eviction_plans.push_back(current_plan);

    }
  }

  sort(eviction_plans.begin(), eviction_plans.end(),
       [](const EvictionPlanInfo& plan_A, const EvictionPlanInfo &plan_B) {
    return plan_A.heat_sum < plan_B.heat_sum;
  });

  int plans_to_execute = std::min(model_count_margin - newly_allocated_count,
                                  (int)eviction_plans.size());

  for (int i = 0; i < plans_to_execute; i++) {


    auto& current_plan = eviction_plans[i];
    auto& instance_name = current_plan.instance_name;
    auto& models_to_evict = current_plan.models_to_evict;

    newly_allocated_count += 1;
    newly_allocated_instances.push_back(instance_name);

    {
      auto model_mgr = get_model_instance_mgr(model_id);

      // early mark as ALLOCATED, for race conditions
      // (multiple entrance in allocate_instance_for_model)
      model_mgr->set_model_state(instance_name, ModelState::ALLOCATED);

      for (const auto& model_to_evict : current_plan.models_to_evict) {
        auto evict_mgr = get_model_instance_mgr(model_to_evict);
        evict_mgr->set_model_state(instance_name, ModelState::DRAINING);
      }
    }

    LOG(INFO) << "Preparing to wake up model " << model_id
              << " on instance " << instance_name;

    wakeup_threads.emplace_back([this, model_id, model_size_bytes, instance_name, models_to_evict]() {
      std::vector<std::thread> sleep_threads;

      for (const auto& model_to_sleep : models_to_evict) {

        LOG(INFO) << "Preparing to wake up model " << model_id
                  << " on instance " << instance_name
                  << ", sending sleep to model " << model_to_sleep;

        sleep_threads.emplace_back([this, instance_name, model_to_sleep]() {
          std::unique_lock<std::mutex> request_metrics_lock(request_metrics_mutex_);
          auto instance_it = request_metrics_.find(instance_name);
          if (instance_it == request_metrics_.end()) {
            LOG(ERROR) << "Failed to find request metrics for instance "
                      << instance_name << " during eviction.";
            return;
          }
          auto model_it = instance_it->second.model_metrics.find(model_to_sleep);
          if (model_it == instance_it->second.model_metrics.end()) {
            LOG(ERROR) << "Failed to find request metrics for model "
                      << model_to_sleep << " on instance "
                      << instance_name << " during eviction.";
            return;
          }

          auto& metrics = model_it->second;
          metrics.cv_idle.wait(request_metrics_lock, [&metrics]() {
            return metrics.prefill_request_num == 0 && metrics.decode_request_num == 0;
          });

          request_metrics_lock.unlock();

          // Double-check can_sleep() to handle D2D lock race condition
          // (a D2D lock could have been acquired after eviction planning)
          auto evict_mgr = get_model_instance_mgr(model_to_sleep);
          if (evict_mgr && !evict_mgr->can_sleep(instance_name)) {
            LOG(WARNING) << "Model " << model_to_sleep << " on " << instance_name
                         << " became D2D locked, skipping sleep";
            return;
          }

          send_model_sleep(instance_name, model_to_sleep);

        });

      }

      for (auto& thread : sleep_threads) {
        if (thread.joinable()) {
          thread.join();
        }
      }

      // Locally deduct free pages for the model being woken up
      deduct_free_pages(instance_name, model_size_bytes);

      send_model_wakeup(instance_name, model_id, /*memory_increased_in_advance*/ true);

    });

  }

  for (auto& thread : wakeup_threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  return newly_allocated_instances;

}

// Select models to evict out >= required_space, meanwhile minimizing sum(heat)
// and considering memory fragmentation when XTensor info is available
EvictionPlanInfo InstanceMgr::select_eviction_candidates(const std::string& instance_name,
                                                         double required_space) {

  // 1. Try to get XTensor info for fragmentation-aware eviction
  std::optional<InstanceXTensorInfo> xtensor_info;
  {
    std::lock_guard<std::mutex> lock(xtensor_info_mutex_);
    auto it = instance_xtensor_infos_.find(instance_name);
    if (it != instance_xtensor_infos_.end() && it->second.is_valid) {
      xtensor_info = it->second;
    }
  }

  // 2. Get all awake models on this instance that can be evicted (not D2D locked)
  std::vector<std::string> awake_models;

  // Iterate all models to check status on this instance
  std::shared_lock<std::shared_mutex> mgr_lock(model_instance_mgr_mutex_);
  for (const auto& pair : model_instance_mgrs_) {
    const std::string& model_id = pair.first;
    auto model_mgr = pair.second;

    ModelState state = model_mgr->get_model_state(instance_name);

    if (state == ModelState::WAKEUP) {
      // Skip if this model on this instance is D2D locked
      if (!model_mgr->can_sleep(instance_name)) {
        LOG(INFO) << "Model " << model_id << " on " << instance_name
                  << " is D2D locked, skipping for eviction";
        continue;
      }

      if (model_mgr->get_wakeup_count() > 1) {
        awake_models.push_back(model_id);
      } else if (model_mgr->get_wakeup_count() == 1) {
        // if this is the only instance for this model, only evict if heat is 0
        if (model_mgr->get_model_heat() == 0) {
          awake_models.push_back(model_id);
        }
      }
    }
  }
  mgr_lock.unlock();

  // 3. Take snapshot of the current model heats
  std::vector<uint64_t> awake_model_heats;
  for (const auto& model_id : awake_models) {
    auto model_mgr = get_model_instance_mgr(model_id);
    awake_model_heats.push_back(model_mgr->get_model_heat());
  }

  // 4. Exhaustive search for the optimal eviction set
  uint64_t required_bytes = static_cast<uint64_t>(required_space * 1024 * 1024 * 1024);
  uint64_t min_sum_heat = std::numeric_limits<uint64_t>::max();
  uint64_t best_contiguous_space = 0;
  size_t best_subset = 0;

  for (size_t subset = 1; subset < (1ULL << awake_models.size()); ++subset) {
    uint64_t current_sum_space_bytes = 0;
    uint64_t current_sum_heat = 0;
    std::vector<std::string> current_evict_models;

    for (size_t i = 0; i < awake_models.size(); ++i) {
      if (subset & (1ULL << i)) {
        current_sum_space_bytes += get_model_size_bytes(awake_models[i]);
        current_sum_heat += awake_model_heats[i];
        current_evict_models.push_back(awake_models[i]);
      }
    }

    // Basic condition: freed space must be sufficient
    if (current_sum_space_bytes < required_bytes) {
      continue;
    }

    // If we have XTensor info, check if contiguous space is sufficient
    if (xtensor_info.has_value()) {
      uint64_t contiguous_space = compute_max_contiguous_free_space(
          xtensor_info.value(),
          current_evict_models,
          static_cast<uint64_t>(kMaxInstanceMemoryGB * 1024 * 1024 * 1024));

      if (contiguous_space < required_bytes) {
        continue;  // Contiguous space not enough, skip this plan
      }

      // Optimization goal: 1. Satisfy contiguous space requirement 2. Minimize heat
      // If heat is the same, prefer the plan with larger contiguous space
      if (current_sum_heat < min_sum_heat ||
          (current_sum_heat == min_sum_heat &&
           contiguous_space > best_contiguous_space)) {
        min_sum_heat = current_sum_heat;
        best_contiguous_space = contiguous_space;
        best_subset = subset;
      }
    } else {
      // No XTensor info, fallback to original logic (heat-only)
      if (current_sum_heat < min_sum_heat) {
        min_sum_heat = current_sum_heat;
        best_subset = subset;
      }
    }
  }

  // 5. Build result
  if (best_subset == 0) {
    return EvictionPlanInfo();  // Cannot find a valid plan
  }

  EvictionPlanInfo best_plan;
  best_plan.instance_name = instance_name;
  best_plan.heat_sum = min_sum_heat;
  for (size_t i = 0; i < awake_models.size(); ++i) {
    if (best_subset & (1ULL << i)) {
      best_plan.models_to_evict.push_back(awake_models[i]);
    }
  }

  return best_plan;
}

// Compute the maximum contiguous free space after evicting specified models
uint64_t InstanceMgr::compute_max_contiguous_free_space(
    const InstanceXTensorInfo& xtensor_info,
    const std::vector<std::string>& models_to_evict,
    uint64_t total_memory_bytes) {

  // Collect all remaining model segments (models NOT being evicted)
  std::vector<WeightSegment> remaining_segments;
  std::unordered_set<std::string> evict_set(
      models_to_evict.begin(), models_to_evict.end());

  for (const auto& [model_id, segments] : xtensor_info.model_weight_segments) {
    if (evict_set.find(model_id) == evict_set.end()) {
      for (const auto& seg : segments) {
        remaining_segments.push_back(seg);
      }
    }
  }

  // Sort by offset
  std::sort(remaining_segments.begin(), remaining_segments.end(),
            [](const WeightSegment& a, const WeightSegment& b) {
              return a.offset < b.offset;
            });

  // Compute all free gaps, find the maximum
  uint64_t max_free = 0;

  // If no remaining segments, the entire space is free
  if (remaining_segments.empty()) {
    return total_memory_bytes;
  }

  // Free space before the first segment
  max_free = std::max(max_free, remaining_segments[0].offset);

  // Free space between adjacent segments
  for (size_t i = 1; i < remaining_segments.size(); ++i) {
    uint64_t gap = remaining_segments[i].offset - remaining_segments[i-1].end();
    max_free = std::max(max_free, gap);
  }

  // Free space after the last segment
  max_free = std::max(max_free,
                      total_memory_bytes - remaining_segments.back().end());

  return max_free;
}

void InstanceMgr::auto_scaling() {

  LOG(INFO) << "~~ Auto scaling disabled.";
  return;

  // Part 1: Distribute instances among models (Scaling Up/Down)
  const std::vector<int> rank_targets = {4, 3}; // Target counts for Top 1, Top 2, etc.

  struct ModelHeat {
    std::string id;
    int64_t heat;
  };
  std::vector<ModelHeat> sorted_models;

  {
    std::shared_lock<std::shared_mutex> mgr_lock(model_instance_mgr_mutex_);
    for (const auto& pair : model_instance_mgrs_) {
      sorted_models.push_back({pair.first, pair.second->get_model_heat()});
      LOG(INFO) << "Model " << pair.first << ": Heat = " << pair.second->get_model_heat();
    }
  }

  std::sort(sorted_models.begin(), sorted_models.end(),
            [](const ModelHeat& a, const ModelHeat& b) {
              return a.heat > b.heat;
            });

  for (size_t i = 0; i < sorted_models.size(); ++i) {
    if (i >= rank_targets.size()) break;

    const auto& model_info = sorted_models[i];
    if (model_info.heat == 0) continue; // Skip if no heat

    int target_count = rank_targets[i];
    auto model_mgr = get_model_instance_mgr(model_info.id);
    int count_margin = target_count - model_mgr->get_allocation_count();

    if (count_margin > 0) {
      LOG(INFO) << "Auto scaling: model " << model_info.id << " (Rank " << i + 1
                << ", Heat " << model_info.heat << ") needs " << count_margin << " more instances.";
      auto instance_names = allocate_instance_for_model(model_info.id, target_count);
      
      if (!instance_names.empty()) {
        LOG(INFO) << "Allocated " << instance_names.size() << " instances for " << model_info.id << ":";
        for (const auto& name : instance_names) {
          LOG(INFO) << "  " << name;
        }
      } else {
        LOG(ERROR) << "Failed to allocate instances for " << model_info.id;
      }
    } else {
      LOG(INFO) << "Model " << model_info.id << " (Rank " << i + 1 << ") satisfied with "
                << model_mgr->get_allocation_count() << " instances (Target: " << target_count << ").";
    }
  }

  // Part 2: Internal Prefill/Decode Instance Distribution (PD Separation)
  {
    std::lock_guard<std::mutex> lock(latency_metrics_mutex_);
    std::shared_lock<std::shared_mutex> mgr_lock(model_instance_mgr_mutex_);
    for (const auto& pair : model_instance_mgrs_) {
      pair.second->auto_flipping(latency_metrics_);
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

  // Find a suitable source instance (not the target itself)
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
      LOG(INFO) << "No valid XTensor info for source instance " << source_instance_name
                << ", skipping for D2D";
      continue;
    }

    // Check if the source has weight segments for this model
    auto seg_it = xtensor_info->model_weight_segments.find(model_id);
    if (seg_it == xtensor_info->model_weight_segments.end() ||
        seg_it->second.empty()) {
      LOG(INFO) << "No weight segments for model " << model_id
                << " on source instance " << source_instance_name;
      continue;
    }

    // Get P2P addresses for D2D weight transfer
    // P2P addresses are mooncake transfer engine addresses (different from device_addrs)
    std::vector<std::string> p2p_addrs;
    if (!xtensor_info->p2p_addrs.empty()) {
      p2p_addrs = xtensor_info->p2p_addrs;
    } else {
      // Fallback: try InstanceMetaInfo.p2p_addrs from registration
      std::shared_lock<std::shared_mutex> lock(inst_mutex_);
      auto meta_it = instances_.find(source_instance_name);
      if (meta_it != instances_.end()) {
        p2p_addrs = meta_it->second.p2p_addrs;
      }
    }

    if (p2p_addrs.empty()) {
      LOG(INFO) << "No P2P addresses for source instance " << source_instance_name
                << ", skipping for D2D";
      continue;
    }

    // Found a suitable source - build D2DWakeupInfo
    D2DWakeupInfo d2d_info;
    d2d_info.source_instance_name = source_instance_name;
    d2d_info.remote_addrs = p2p_addrs;

    // For each remote addr, add the weight segments
    // Currently assuming all workers have the same weight segments
    for (size_t i = 0; i < p2p_addrs.size(); ++i) {
      d2d_info.src_weight_segments.push_back(seg_it->second);
    }

    selected_source = source_instance_name;

    LOG(INFO) << "Found D2D source instance " << source_instance_name
              << " for model " << model_id
              << " with " << d2d_info.remote_addrs.size() << " remote addrs"
              << " and " << seg_it->second.size() << " weight segments";

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

  // No suitable source found - unlock all instances
  model_mgr->release_d2d_locks(locked_instances);

  LOG(INFO) << "No suitable D2D source found for model " << model_id
            << ", will use H2D wakeup";
  return std::nullopt;
}

}  // namespace xllm_service
