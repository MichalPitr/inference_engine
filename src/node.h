#ifndef NODE_H
#define NODE_H

#include <optional>
#include <string>

#include "attribute.h"
#include "onnx-ml.pb.h"
#include "optypes.h"
#include "tensor.h"

class Node {
   public:
    Node(const std::string &name, const OpType optype);
    Node(const onnx::NodeProto &nodeProto);

    const std::string &getName() const;
    OpType getOpType() const;
    const std::vector<std::string> &getInputs() const;
    const std::vector<std::string> &getOutputs() const;
    const std::unordered_map<std::string, Attribute> &getAttributes() const;
    template <typename T>
    std::optional<T> getAttribute(const std::string &name) const;

    void addInput(std::string input);
    void addOutput(std::string output);

   private:
    std::string name;
    OpType opType;
    // Sorted list of inputs as expected the by the corresponding opType.
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::unordered_map<std::string, Attribute> attributes;
};

#endif