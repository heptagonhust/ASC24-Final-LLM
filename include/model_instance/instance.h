#pragma once
#include "model_instance/config.h"
#include <memory>

class Instance {
public:
    Instance(InstanceParams instanceParams);

    void run();
private:
    InstanceParams instanceParams_;
    std::shared_ptr<InstanceConfig> config_;
    std::shared_ptr<ExecutorServer> executorServer_;
    std::shared_ptr<Recorder> recorder_;
};
