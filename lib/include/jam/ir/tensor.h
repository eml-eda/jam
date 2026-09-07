#ifndef __JAM_IR_TENSOR_H__
#define __JAM_IR_TENSOR_H__

#include <jam/ir/config.h>
#include <jam/types.h>

typedef enum {
    JAM_IR_DATA_TYPE_F32,
    JAM_IR_DATA_TYPE_I8,
    JAM_IR_DATA_TYPE_I16,
    JAM_IR_DATA_TYPE_I32,
} jam_ir_data_type_t;

typedef struct {
    uint8_t num_dims;
    uint8_t shape[__JAM_MAX_NUM_DIMS__];
} jam_ir_shape_t;

typedef struct {
    jam_ir_shape_t shape;
    uintptr_t pt;
    uint8_t valid;
} jam_ir_tile_t;

// a tensor is expected to live in l3 or l2 at the start
typedef struct {
    jam_ir_tile_t l3_tile;
    jam_ir_tile_t l2_tile;
    jam_ir_tile_t l1_tile;
    uint8_t data_type;
} jam_ir_tensor_t;

#endif /* __JAM_IR_TENSOR_H__ */