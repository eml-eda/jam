#ifndef __JAM_SESSION_H__
#define __JAM_SESSION_H__

// the jam session should include all the necessary parts to run a jam app
typedef struct {
    jam_app *app;
    jam_device *device;
    jam_status status;
} jam_session;

void load_jam_ir(jam_session *session, const void* ir_data, size_t ir_size);
void compile_jam_session(jam_session *session);

#endif /* __JAM_SESSION_H__ */