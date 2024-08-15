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
    nodes_[nodeName] = std::move(node);

    Node *nodePtr = nodes_[nodeName].get();
    adjList_[nodePtr] = std::vector<Node *>{};

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

void Graph::replaceNode(Node* oldNode, std::unique_ptr<Node> newNode) {
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
    if (sortedNodes_.size() > 0) {
        return sortedNodes_;
    }

    std::unordered_set<Node *> visited;
    std::stack<Node *> stack;

    std::vector<Node *> input_nodes;
    for (auto &nodePair : nodes_)
    {
        for (auto &node_input : nodePair.second.get()->getInputs())
        {
            for (auto &graph_input : inputs_)
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

    for (auto node : input_nodes)
    {
        topologicalSortUtil(node, visited, stack);
    }

    std::vector<Node *> sortedNodes;
    while (!stack.empty())
    {
        sortedNodes.push_back(stack.top());
        stack.pop();
    }

    // Caching
    sortedNodes_ = sortedNodes;
    return sortedNodes;
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
