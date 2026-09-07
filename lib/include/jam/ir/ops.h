#ifndef __JAM_IR_OPS_H__
#define __JAM_IR_OPS_H__

#include <jam/ir/config.h>
#include <jam/types.h>

typedef enum {
    JAM_IR_OP_CONV,
    JAM_IR_OP_GEMM,
    JAM_IR_OP_MATMUL,
    JAM_IR_OP_AVG_POOL,
    JAM_IR_OP_MAX_POOL,
    JAM_IR_OP_TRANSPOSE,
    JAM_IR_OP_ELEMENTWISE_ADD,
    JAM_IR_OP_ELEMENTWISE_MUL,
} jam_ir_op_type_t;

typedef struct {
    uint8_t kernel_size[__JAM_MAX_NUM_DIMS_SPAT__];
    uint8_t stride[__JAM_MAX_NUM_DIMS_SPAT__];
    uint8_t padding[__JAM_MAX_NUM_DIMS_SPAT__][2];
    uint8_t num_dims_spat;
} spatial_attributes_t;

typedef struct {
    spatial_attributes_t spatial;
} pool_attributes_t;

typedef struct {
    spatial_attributes_t spatial;
    // uint8_t groups;
} conv_attributes_t;

typedef struct {
    uint8_t axis;
} transpose_attributes_t;

typedef struct {
    uint8_t trans_a;
    uint8_t trans_b;
} gemm_attributes_t;

typedef struct {
    uint8_t type;
    union {
        pool_attributes_t pool;
        conv_attributes_t conv;
        transpose_attributes_t transpose;
        gemm_attributes_t gemm;
    } attributes;
    uint8_t elementwise_type[__JAM_MAX_NUM_ELEMENTWISE_OPS__];
    uint8_t activation_type;
    uint8_t input_refs[__JAM_MAX_NUM_INPUTS__];
    uint8_t num_inputs;
    uint8_t output_refs[__JAM_MAX_NUM_OUTPUTS__];
    uint8_t num_outputs;
} jam_ir_op_t;

#endif /* __JAM_IR_OPS_H__ */