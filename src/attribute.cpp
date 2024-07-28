#include "attribute.h"

Attribute::Attribute(const onnx::AttributeProto &attrProto)
    : name(attrProto.name())
{
    switch (attrProto.type())
    {
    case onnx::AttributeProto::INT:
        value = attrProto.i();
        break;
    case onnx::AttributeProto::FLOAT:
        value = attrProto.f();
        break;
    default:
        throw std::runtime_error("Unsupported attribute type");
    }
}

const std::string &Attribute::getName() const { return name; }
const Attribute::AttributeValue &Attribute::getValue() const
{
    return value;
}