#pragma once
#include <memory>

#include "model_instance/config.h"
#include "multinodes/multinodes_server.h"
#include "rpc/client.h"
class Instance {
public:
    Instance(InstanceParams instanceParams,std::string client_id);
    void run();
    std::vector<texec::Request> getRequests(Sequences seqs) const;

private:
    InstanceParams instanceParams_;
    std::shared_ptr<InstanceConfig> config_;
    std::shared_ptr<ExecutorServer> executorServer_;
    std::shared_ptr<Recorder> recorder_;
    std::string client_id_;
};
