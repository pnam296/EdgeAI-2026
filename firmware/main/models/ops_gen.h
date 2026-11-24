#pragma once
/* SINH TU DONG boi tools/tfl_utils.py (dung sua tay). Model: det_model.cc, textcnn_model.cc */
#define RESOLVER_SIZE 16
#define REGISTER_OPS(r) do { \
    (r).AddAdd(); \
    (r).AddConcatenation(); \
    (r).AddConv2D(); \
    (r).AddDepthwiseConv2D(); \
    (r).AddFullyConnected(); \
    (r).AddGather(); \
    (r).AddLogistic(); \
    (r).AddMaxPool2D(); \
    (r).AddMul(); \
    (r).AddPad(); \
    (r).AddQuantize(); \
    (r).AddReduceMax(); \
    (r).AddReshape(); \
    (r).AddResizeNearestNeighbor(); \
    (r).AddSoftmax(); \
    (r).AddSplit(); \
} while (0)
