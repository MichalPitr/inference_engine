#ifndef GRAPH_H
#define GRAPH_H

#include <string>
#include <vector>
#include <stack>

#include "node.h"
#include "onnx-ml.pb.h"

class Graph
{
public:
    Graph();
    Graph(const onnx::GraphProto &graphProto);

    void addNode(std::unique_ptr<Node> node);

    const std::string &getInputName(std::size_t index) const;
    const std::string &getOutputName(std::size_t index) const;
    std::vector<Node *> getTopologicallySortedNodes() const;

private:
    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;
    std::unordered_map<std::string, std::unique_ptr<Node>> nodes_;

    void topologicalSortUtil(const Node *node, std::unordered_set<const Node *> &visited, std::stack<const Node *> &stack) const;
};

#endif