#include "model_instance/instance.h"
#include "model_instance/config.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>

Instance::Instance(InstanceParams instanceParams) 
    : instanceParams_(instanceParams) 
{
    config_ = InstanceConfig::from_params(instanceParams_);
    recorder_ = config_->getRecorder();
    executorServer_ = config_->getExecutorServer(recorder_);
}


void Instance::run()
{
    // Warm up
    {
        int warm_up_iterations = instanceParams_.serverParams.warm_up_iterations;
        auto reqs = config_->getRequests(warm_up_iterations);
        executorServer_->enqueue(std::move(reqs), true);
        executorServer_->waitForResponses(reqs.size(), true);
    }
    recorder_->initialize();
    {
        auto reqs = config_->getRequests();
        executorServer_->enqueue(std::move(reqs));
        executorServer_->waitForResponses(reqs.size());
    }
    recorder_->finalize();
    recorder_->calculateMetrics();
    recorder_->report();
    recorder_->writeOpMetricsToCsv();
}

void Instance::writeResultsToJson(const fs::path& outputPath) const
{
    auto results = executorServer_->getResults();
    nlohmann::json j;
    for (auto& [reqId, result] : results)
    {
        j.push_back({result.outputTokenIds[0]});
    }
    std::ofstream outputFile(outputPath, std::ios::out);
    outputFile << j << std::endl;
}