#pragma once
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <map>

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"

#include "model_instance/recorder.h"

using namespace tensorrt_llm::batch_manager;

namespace fs = std::filesystem;
namespace texec = tensorrt_llm::executor;

class ExecutorServer {
public:
    ExecutorServer(
        fs::path engineDir,
        texec::ExecutorConfig const& executorConfig,
        std::shared_ptr<Recorder> recorder,
        std::chrono::milliseconds waitSleep
    );

    ~ExecutorServer() {}

    void enqueue(std::vector<texec::Request> requests, bool warmup = false);

    void waitForResponses(std::optional<SizeType> numRequests,
                          bool warmup = false);
        
    std::map<texec::IdType, texec::Result> getResults() { return results_; }

    void shutdown() { executor_->shutdown(); }

private:
    std::shared_ptr<texec::Executor> executor_;
    std::shared_ptr<Recorder> recorder_;
    std::chrono::milliseconds waitSleep_;
    std::atomic<uint64_t> activeCount_;
    std::map<texec::IdType, texec::Result> results_;
}; // class ExecutorServer

