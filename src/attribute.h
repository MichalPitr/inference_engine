#ifndef ATTRIBUTE_H
#define ATTRIBUTE_H

#include <variant>
#include <vector>
#include <string>
#include "onnx-ml.pb.h"

class Attribute
{
public:
    using AttributeValue = std::variant<int64_t, float, std::vector<int64_t>>;

    Attribute(const onnx::AttributeProto &attrProto);

    const std::string &getName() const;
    const AttributeValue &getValue() const;

private:
    std::string name;
    AttributeValue value;
};

#endif // ATTRIBUTE_H