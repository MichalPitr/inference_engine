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
    void addNode(Node*);
    void addInput(std::string);
    void addOutput(std::string);

private:
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<Node*> nodes;
};

#endif