#include "graph.h"

Graph::Graph() {}

Graph::Graph(const onnx::GraphProto &graphProto)
{
    for (const auto &nodeProto : graphProto.node())
    {
        addNode(std::make_unique<Node>(nodeProto));
    }
    for (const auto &inputProto : graphProto.input())
    {
        inputs_.push_back(inputProto.name());
    }
    for (const auto &outputProto : graphProto.output())
    {
        outputs_.push_back(outputProto.name());
    }
}

void Graph::addNode(std::unique_ptr<Node> node)
{
    nodes_[node->getName()] = std::move(node);
}

const std::string &Graph::getInputName(std::size_t index) const
{
    return inputs_.at(index);
}

const std::string &Graph::getOutputName(std::size_t index) const
{
    return outputs_.at(index);
}

std::vector<Node *> Graph::getTopologicallySortedNodes() const
{
    std::unordered_set<const Node *> visited;
    std::stack<const Node *> stack;

    for (const auto &nodePair : nodes_)
    {
        if (visited.find(nodePair.second.get()) == visited.end())
        {
            topologicalSortUtil(nodePair.second.get(), visited, stack);
        }
    }

    std::vector<Node *> sortedNodes;
    while (!stack.empty())
    {
        sortedNodes.push_back(const_cast<Node *>(stack.top()));
        stack.pop();
    }

    return sortedNodes;
}

void Graph::topologicalSortUtil(const Node *node, std::unordered_set<const Node *> &visited, std::stack<const Node *> &stack) const
{
    visited.insert(node);

    for (const auto &input_name : node->getInputs())
    {
        auto it = nodes_.find(input_name);
        if (it != nodes_.end() && visited.find(it->second.get()) == visited.end())
        {
            topologicalSortUtil(it->second.get(), visited, stack);
        }
    }

    stack.push(node);
}
