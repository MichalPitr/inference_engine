#include "onnx_helper.h"

template <typename T>
std::tuple<bool, T> getAttr(const onnx::NodeProto& node, const std::string& attrName) {
    for (const auto& attr : node.attribute()) {
        if (attr.name() == attrName) {
            if (std::is_same<T, float>::value) {
                return {true, attr.f()};
            } else if (std::is_same<T, int>::value) {
                return {true, attr.i()};
            }
            // Add more cases for other types if needed
        }
    }
    return {false, {}};  // Attribute not found
}

template std::tuple<bool, float> getAttr(const onnx::NodeProto& node, const std::string& attrName);
template std::tuple<bool, int> getAttr(const onnx::NodeProto& node, const std::string& attrName);
