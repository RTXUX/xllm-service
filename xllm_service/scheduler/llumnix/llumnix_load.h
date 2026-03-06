#pragma once

#include <string>

namespace xllm_service {

// Dispatch policy types
enum class LlumnixDispatchPolicy {
  LOAD,       // Sort by load metric, pick from top-K randomly
  BALANCED,   // Pick instance with fewest total requests
  QUEUE,      // Pick instance with fewest waiting requests
  ROUND_ROBIN // Simple round-robin
};

// Migration policy types
enum class LlumnixMigrationPolicy {
  BALANCED,  // Pair high-src with low-dst, verify improvement
  DEFRAG     // Aggressive pairing without constraint checks
};

// Load metric types
enum class LlumnixLoadMetric {
  KV_BLOCKS_RATIO,   // gpu_cache_usage_perc from heartbeat
  REMAINING_STEPS    // Approximation based on request count
};

inline LlumnixDispatchPolicy parse_dispatch_policy(const std::string& s) {
  if (s == "balanced") return LlumnixDispatchPolicy::BALANCED;
  if (s == "queue") return LlumnixDispatchPolicy::QUEUE;
  if (s == "rr") return LlumnixDispatchPolicy::ROUND_ROBIN;
  return LlumnixDispatchPolicy::LOAD;  // default
}

inline LlumnixMigrationPolicy parse_migration_policy(const std::string& s) {
  if (s == "defrag") return LlumnixMigrationPolicy::DEFRAG;
  return LlumnixMigrationPolicy::BALANCED;  // default
}

inline LlumnixLoadMetric parse_load_metric(const std::string& s) {
  if (s == "remaining_steps") return LlumnixLoadMetric::REMAINING_STEPS;
  return LlumnixLoadMetric::KV_BLOCKS_RATIO;  // default
}

}  // namespace xllm_service
