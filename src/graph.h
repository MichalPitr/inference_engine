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
    struct NodeInfo {
        std::unique_ptr<Node> node;
        std::vector<Node*> children;
        std::vector<Node*> parents;
    };


    Graph() = default;
    Graph(const onnx::GraphProto &graphProto);

    const std::string &getInputName(std::size_t index) const;
    const std::string &getOutputName(std::size_t index) const;
    void printGraph() const;
    
    std::vector<Node *> getTopologicallySortedNodes();
    void addNode(std::unique_ptr<Node> node);
    void replaceNode(Node* oldNode, std::unique_ptr<Node> newNode);

private:
    void updateEdges(Node* node);
    void addIncomingEdges(Node* node);
    bool isInputNode(Node* node) const;
    void addOutgoingEdges(Node* node);
    void topologicalSortUtil(Node *node, std::unordered_set<Node *> &visited, std::stack<Node *> &stack);
    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;
    std::unordered_map<std::string, NodeInfo> nodeMap_;
    std::vector<Node*> sortedNodes_;
};

#endif