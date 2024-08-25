#include "graph.h"

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
    // printGraph();
}

void Graph::addNode(std::unique_ptr<Node> node)
{
    std::string nodeName = node->getName();
    Node *nodePtr = node.get();
    nodes_[nodeName] = std::move(node);
    adjList_[nodePtr] = {};

    updateEdges(nodePtr);
}

void Graph::updateEdges(Node *node)
{
    addIncomingEdges(node);
    addOutgoingEdges(node);
}

void Graph::addIncomingEdges(Node *node)
{
    for (const auto &inputName : node->getInputs())
    {
        for (const auto &[_, existingNode] : nodes_)
        {
            if (std::find(existingNode->getOutputs().begin(), existingNode->getOutputs().end(), inputName) != existingNode->getOutputs().end())
            {
                adjList_[existingNode.get()].push_back(node);
            }
        }
    }
}

void Graph::addOutgoingEdges(Node *node)
{
    for (const auto &outputName : node->getOutputs())
    {
        for (const auto &[_, existingNode] : nodes_)
        {
            if (std::find(existingNode->getInputs().begin(), existingNode->getInputs().end(), outputName) != existingNode->getInputs().end())
            {
                adjList_[node].push_back(existingNode.get());
            }
        }
    }
}

void Graph::replaceNode(Node *oldNode, std::unique_ptr<Node> newNode)
{
    std::string name = oldNode->getName();
    auto cp = adjList_[oldNode];
    nodes_[name] = std::move(newNode);
    adjList_[nodes_[name].get()] = cp;
}

const std::string &Graph::getInputName(std::size_t index) const
{
    return inputs_.at(index);
}

const std::string &Graph::getOutputName(std::size_t index) const
{
    return outputs_.at(index);
}

std::vector<Node *> Graph::getTopologicallySortedNodes()
{
    if (!sortedNodes_.empty())
    {
        return sortedNodes_;
    }

    std::unordered_set<Node *> visited;
    std::stack<Node *> stack;

    for (const auto &[_, node] : nodes_)
    {
        if (isInputNode(node.get()))
        {
            topologicalSortUtil(node.get(), visited, stack);
        }
    }

    sortedNodes_.reserve(stack.size());
    while (!stack.empty())
    {
        sortedNodes_.push_back(stack.top());
        stack.pop();
    }

    return sortedNodes_;
}

bool Graph::isInputNode(Node *node) const
{
    return std::any_of(node->getInputs().begin(), node->getInputs().end(),
                       [this](const std::string &input)
                       {
                           return std::find(inputs_.begin(), inputs_.end(), input) != inputs_.end();
                       });
}

void Graph::topologicalSortUtil(Node *node, std::unordered_set<Node *> &visited, std::stack<Node *> &stack)
{
    visited.insert(node);

    std::vector<Node *> children = adjList_.at(node);
    if (children.size() == 0)
    {
        stack.push(node);
        return;
    }

    for (auto child : children)
    {
        if (visited.find(child) == visited.end())
        {
            topologicalSortUtil(child, visited, stack);
        }
    }

    stack.push(node);
}

void Graph::printGraph() const
{
    for (const auto &keyVal : adjList_)
    {
        std::cout << "Node " << keyVal.first->getName() << ": \n";
        for (const auto &node : keyVal.second)
        {
            std::cout << "    child " << node->getName() << "\n";
        }
    }
}
