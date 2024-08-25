#ifndef ONNX_HELPER_H
#define ONNX_HELPER_H

#include <tuple>

#include "onnx-ml.pb.h"

template <typename T>
std::tuple<bool, T> getAttr(const onnx::NodeProto& node,
                            const std::string& attrName);

#endif