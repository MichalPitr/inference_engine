#include "attribute.h"

Attribute::Attribute(const onnx::AttributeProto &attrProto)
    : name(attrProto.name()) {
    switch(attrProto.type()) {
        case onnx::AttributeProto::INT: {
            value = attrProto.i();
            break;
        }
        case onnx::AttributeProto::FLOAT: {
            value = attrProto.f();
            break;
        }
        case onnx::AttributeProto::INTS: {
            std::vector<int64_t> ints(attrProto.ints_size());
            for(int i = 0; i < attrProto.ints_size(); ++i) {
                ints[i] = attrProto.ints(i);
            }
            value = ints;
            break;
        }
        default:
            throw std::runtime_error("Unsupported attribute type" +
                                     std::to_string(attrProto.type()));
    }
}

const std::string &Attribute::getName() const { return name; }
const Attribute::AttributeValue &Attribute::getValue() const { return value; }