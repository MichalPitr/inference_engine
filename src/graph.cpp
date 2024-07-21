#include "graph.h"

Graph::Graph() {}

void Graph::addNode(std::unique_ptr<Node> node) {
    nodes_.emplace(node->getName(), std::move(node));
}

const Node& Graph::getNode(const std::string& nodeName) const {
    // TODO: add error handling.
    return *(nodes_.at(nodeName)); 
}

void Graph::addInput(std::string input) {
    inputs.push_back(input);
}

void Graph::addOutput(std::string output) {
    outputs.push_back(output);
}