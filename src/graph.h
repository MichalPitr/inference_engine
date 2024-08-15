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
    Graph() = default;
    Graph(const onnx::GraphProto &graphProto);

    const std::string &getInputName(std::size_t index) const;
    const std::string &getOutputName(std::size_t index) const;
    void printGraph() const;
    
    std::vector<Node *> getTopologicallySortedNodes();
    void addNode(std::unique_ptr<Node> node);
    void replaceNode(Node* oldNode, std::unique_ptr<Node> newNode);

private:
    void topologicalSortUtil(Node *node, std::unordered_set<Node *> &visited, std::stack<Node *> &stack);
    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;
    std::unordered_map<std::string, std::unique_ptr<Node>> nodes_;
    std::unordered_map<Node*, std::vector<Node*>> adjList_;
    std::vector<Node*> sortedNodes_;
};

#endif