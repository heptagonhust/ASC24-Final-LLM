#pragma once
#include <atomic>
#include <cstdint>
#include <filesystem>

#include "tensorrt_llm/executor/executor.h"

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

    void shutdown() { mExecutor->shutdown(); }

private:
    std::shared_ptr<texec::Executor> mExecutor;
    std::shared_ptr<Recorder> mRecorder;
    std::chrono::milliseconds mWaitSleep;
    std::atomic<uint64_t> mActiveCount;
}; // class ExecutorServer

