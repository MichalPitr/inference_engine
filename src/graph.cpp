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
    // printGraph();
}

void Graph::addNode(std::unique_ptr<Node> node)
{
    std::string nodeName = node->getName();
    nodes_[nodeName] = std::move(node);

    Node *nodePtr = nodes_[nodeName].get();
    adjList_[nodePtr] = std::vector<const Node *>{};

    // Check if node is child of existing nodes.
    for (const auto &inputName : nodePtr->getInputs())
    {
        for (const auto &existingNodePair : nodes_)
        {
            Node *existingNode = existingNodePair.second.get();
            if (std::find(existingNode->getOutputs().begin(), existingNode->getOutputs().end(), inputName) != existingNode->getOutputs().end())
            {
                adjList_[existingNode].push_back(nodePtr);
            }
        }
    }

    // Check if any nodes are children of current node.
    for (const auto &outputName : nodePtr->getOutputs())
    {
        for (const auto &existingNodePair : nodes_)
        {
            Node *existingNode = existingNodePair.second.get();
            if (std::find(existingNode->getInputs().begin(), existingNode->getInputs().end(), outputName) != existingNode->getInputs().end())
            {
                adjList_[nodePtr].push_back(existingNode);
            }
        }
    }
}

const std::string &Graph::getInputName(std::size_t index) const
{
    return inputs_.at(index);
}

const std::string &Graph::getOutputName(std::size_t index) const
{
    return outputs_.at(index);
}

std::vector<const Node *> Graph::getTopologicallySortedNodes()
{
    if (sortedNodes_.size() > 0) {
        return sortedNodes_;
    }

    std::unordered_set<const Node *> visited;
    std::stack<const Node *> stack;

    std::vector<const Node *> input_nodes;
    for (const auto &nodePair : nodes_)
    {
        for (const auto &node_input : nodePair.second.get()->getInputs())
        {
            for (const auto &graph_input : inputs_)
            {
                if (node_input == graph_input)
                {
                    input_nodes.push_back(nodePair.second.get());
                    goto endloop;
                }
            }
        }
    // Once node is added, break out of two nested loops to avoid adding it multiple times if it has multiple root inputs.
    endloop:;
    }

    for (const auto node : input_nodes)
    {
        topologicalSortUtil(node, visited, stack);
    }

    std::vector<const Node *> sortedNodes;
    while (!stack.empty())
    {
        sortedNodes.push_back(stack.top());
        stack.pop();
    }

    // Caching
    sortedNodes_ = sortedNodes;
    return sortedNodes;
}

void Graph::topologicalSortUtil(const Node *node, std::unordered_set<const Node *> &visited, std::stack<const Node *> &stack) const
{
    visited.insert(node);

    const std::vector<const Node *> children = adjList_.at(node);
    if (children.size() == 0)
    {
        stack.push(node);
        return;
    }

    for (const auto child : children)
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
