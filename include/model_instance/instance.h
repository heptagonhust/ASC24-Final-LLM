#pragma once
#include <memory>

#include "model_instance/config.h"
#include "sequences/sequences.h"

class Instance {
public:
    Instance(InstanceParams instanceParams);
    void run();
    std::vector<texec::Request> getRequests(Sequences seqs) const;

private:
    InstanceParams instanceParams_;
    std::shared_ptr<InstanceConfig> config_;
    std::shared_ptr<ExecutorServer> executorServer_;
    std::shared_ptr<Recorder> recorder_;
};
