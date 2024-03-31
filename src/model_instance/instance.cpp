#include <memory>
#include <optional>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>

#include "rpc/client.h"

#include "model_instance/instance.h"
#include "model_instance/config.h"
#include "model_instance/config.h"

Instance::Instance(InstanceParams instanceParams)
    : instanceParams_(instanceParams)
{
    config_ = InstanceConfig::from_params(instanceParams_);
    recorder_ = config_->getRecorder();
    executorServer_ = config_->getExecutorServer(recorder_);
}

void Instance::run()
{
    rpc::client client(
        instanceParams_.rpcParams.rpcAddress, 
        instanceParams_.rpcParams.rpcPort
    );
    // Warm up
    {
        Sequences seqs_warmup;
        seqs_warmup.push_back(Sequence{ {2300, 118, 9022, 504, 6390, 5002, 1221, 859, 221, 768, 33532, 5084, 521, 44528, 1310, 859}, 200 });
        seqs_warmup.push_back(Sequence{ {33733, 429, 5865, 20003, 1603, 329, 11007, 859}, 200 });
        auto reqs = getRequests(seqs_warmup);
        executorServer_->enqueue(std::move(reqs), true);
        executorServer_->waitForResponses(reqs.size(), true);
    }
    recorder_->initialize();

    // execute
    std::vector<int32_t> order;
    Sequences seqs = client.call("getseqs").as<Sequences>();
    order.push_back(seqs.at(0).order_id);
    while (1) {
        auto reqs = getRequests(seqs);
        executorServer_->enqueue(std::move(reqs));
        executorServer_->waitForGetReqs(instanceParams_.rpcParams.rpcNseqsThreshold);
        seqs = client.call("getseqs").as<Sequences>();
        if(seqs.size() == 0)
            break;
        order.push_back(seqs.at(0).order_id);
    }
    executorServer_->waitForResponses();
    recorder_->finalize();
    recorder_->calculateMetrics();
    recorder_->writeOpMetricsToCsv();
    if (instanceParams_.loggerParams.showResults) {
        recorder_->report();
    }

    auto results = executorServer_->getResults();
    std::vector<std::vector<int32_t>> outIds;
    for (auto& [reqId, result] : results) {
        outIds.push_back(result.outputTokenIds[0]);
    }
    client.call("outseqs_back",outIds,order);

}

std::vector<texec::Request> Instance::getRequests(Sequences seqs) const {
    auto numSequences = seqs.size();
    std::vector<texec::Request> requests;
    for (int i = 0; i < numSequences; ++i) {
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
