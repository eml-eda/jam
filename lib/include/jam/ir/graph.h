#ifndef __JAM_IR_GRAPH_H__
#define __JAM_IR_GRAPH_H__

#include <jam/types.h>
#include <jam/ir/ops.h>
#include <jam/ir/tensor.h>

typedef struct {
    jam_ir_op_t *ops;
    uint16_t num_ops;
    jam_ir_tensor_t *tensors;
    uint16_t num_tensors;
} graph_t;

#endif /* __JAM_IR_GRAPH_H__ */