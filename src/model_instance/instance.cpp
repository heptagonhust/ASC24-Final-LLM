#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <vector>

#include "model_instance/instance.h"
#include "model_instance/config.h"
#include "model_instance/request.h"
#include "model_instance/config.h"


Instance::Instance(InstanceParams instanceParams) 
    : instanceParams_(instanceParams) 
{
    config_ = InstanceConfig::from_params(instanceParams_);
    recorder_ = config_->getRecorder();
    executorServer_ = config_->getExecutorServer(recorder_);
    seqs_ =  readDatasetFromJson(
        config_->getDatasetPath(),
        config_->getTokenizerPath(),
        instanceParams_.reqParams.maxNumSequences
    );
}

void Instance::run()
{
    // Warm up
    {
        int warm_up_iterations = instanceParams_.serverParams.warm_up_iterations;
        auto reqs = getRequests(warm_up_iterations);
        executorServer_->enqueue(std::move(reqs), true);
        executorServer_->waitForResponses(reqs.size(), true);
    }
    recorder_->initialize();
    {
        auto reqs = getRequests();
        executorServer_->enqueue(std::move(reqs));
        executorServer_->waitForResponses(reqs.size());
    }
    recorder_->finalize();
    recorder_->calculateMetrics();
    recorder_->report();
    recorder_->writeOpMetricsToCsv();
}


std::vector<texec::Request> Instance::getRequests(std::optional<size_t> num) const {
    const auto& seqs = seqs_;
    auto numSequences = seqs.size();

    num = num.has_value() ? std::min(num.value(), numSequences) : numSequences;
    std::vector<texec::Request> requests;
    for (int i = 0; i < num; ++i) {
        requests.push_back(
            texec::Request {
                seqs[i].inputIds,
                seqs[i].outputLen,
                instanceParams_.engineParams.streaming,
                config_->getSamplingConfig(),
                config_->getOutputConfig(),
                instanceParams_.modelParams.eosId,
                instanceParams_.modelParams.padId,
            }
        );
    }
    return requests;
}

void Instance::writeResults() const
{
    writeResultsToJson(
        config_->getOutputPath(), 
        config_->getTokenizerPath(),
        std::move(executorServer_->getResults()));
}