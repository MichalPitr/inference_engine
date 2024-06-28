#include "graph.h"

Graph::Graph() {}

void Graph::addInput(std::string input) {
    inputs.push_back(input);
}

void Graph::addOutput(std::string output) {
    outputs.push_back(output);
}