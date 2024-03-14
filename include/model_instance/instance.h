#pragma once
#include "model_instance/config.h"
#include <memory>

class Instance {
public:
    Instance(InstanceParams instanceParams);
    void run();
    std::vector<texec::Request> getRequests(std::optional<size_t> num = std::nullopt) const;
    void writeResults() const;

private:
    InstanceParams instanceParams_;
    std::shared_ptr<InstanceConfig> config_;
    std::shared_ptr<ExecutorServer> executorServer_;
    std::shared_ptr<Recorder> recorder_;
    Sequences seqs_;
};
