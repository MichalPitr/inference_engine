#ifndef GRAPH_H
#define GRAPH_H

#include <string>
#include <vector>

#include "node.h"

class Graph {
public:
    Graph();
    
    std::vector<Node*> getNodes();
    std::vector<std::string> getInputs();
    std::vector<std::string> getOutputs();
    void addNode(std::unique_ptr<Node> node);
    void addInput(std::string);
    void addOutput(std::string);
    const Node& getNode(const std::string& nodeName) const;

private:
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::unordered_map<std::string, std::unique_ptr<Node>> nodes_;  
};

#endif