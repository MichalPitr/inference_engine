#include "graph.h"

Graph::Graph(const onnx::GraphProto &graphProto)
{
    nodeMap_.reserve(graphProto.node_size());
    for (const auto &nodeProto : graphProto.node())
    {
        addNode(std::make_unique<Node>(nodeProto));
    }

    inputs_.reserve(graphProto.input_size());
    for (const auto &inputProto : graphProto.input())
    {
        inputs_.push_back(inputProto.name());
    }

    outputs_.reserve(graphProto.output_size());
    for (const auto &outputProto : graphProto.output())
    {
        outputs_.push_back(outputProto.name());
    }
}

void Graph::addNode(std::unique_ptr<Node> node)
{
    std::string nodeName = node->getName();
    Node *nodePtr = node.get();
    nodeMap_[nodeName].node = std::move(node);

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
        for (auto &[_, info] : nodeMap_)
        {
            if (std::find(info.node->getOutputs().begin(), info.node->getOutputs().end(), inputName) != info.node->getOutputs().end())
            {
                info.children.push_back(node);
                nodeMap_[node->getName()].parents.push_back(info.node.get());
            }
        }
    }
}

void Graph::addOutgoingEdges(Node *node)
{
    for (const auto &outputName : node->getOutputs())
    {
        for (auto &[_, info] : nodeMap_)
        {
            if (std::find(info.node->getInputs().begin(), info.node->getInputs().end(), outputName) != info.node->getInputs().end())
            {
                nodeMap_[node->getName()].children.push_back(info.node.get());
                info.parents.push_back(node);
            }
        }
    }
}

void Graph::replaceNode(Node *oldNode, std::unique_ptr<Node> newNode)
{
    // Note: This assumes that the new node has the same connections as the old one.
    // If this is not the case, parents/children need to updated manually.
    std::string name = oldNode->getName();
    auto &info = nodeMap_[name];
    info.node = std::move(newNode);
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

    for (const auto &[_, info] : nodeMap_)
    {
        if (info.parents.empty() || isInputNode(info.node.get()))
        {
            topologicalSortUtil(info.node.get(), visited, stack);
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

    for (Node *child : nodeMap_[node->getName()].children)
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
    for (const auto &[name, info] : nodeMap_)
    {
        std::cout << "Node " << name << ": \n";
        for (const auto &child : info.children)
        {
            std::cout << "    child " << child->getName() << "\n";
        }
    }
}