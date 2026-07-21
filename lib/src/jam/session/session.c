#include <jam/session/session.h>

void compile_jam_session(jam_session *session) {
    // annotate the graph nodes with device support information
    annotate_graph_nodes(session->app->graph, session->device);
    // tile nodes trying to utilize compute units minimum grid sizes
    // and leaving the rest of the work to be done by the host
    // (matmul 25x25, and have an accelerator 16x16 and one 8x8, then tile in 16x16 and 8x8, leaving the remaining 1x1 to be done by the host)
    tile_graph_nodes(session->app->graph, session->device);
    // add single compute unit data transfers
    for(int i = 0; i < session->device->num_compute_units; i++)
        add_single_compute_unit_data_transfers(session->app->graph, session->device, i);
    // analyze data dependencies between tiles from different compute units and insert necessary synch barriers
    if(session->device->num_compute_units > 1)
        complete_data_dependency_analysis(session->app->graph, session->device);
    // generate ucode for each compute unit
    for(int i = 0; i < session->device->num_compute_units; i++)
        generate_ucode_for_compute_unit(session->app->graph, session->device, i);
    return;
}