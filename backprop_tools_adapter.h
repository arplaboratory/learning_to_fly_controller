#ifdef __cplusplus
extern "C"
#endif
void backprop_tools_init();
#ifdef __cplusplus
extern "C"
#endif
float backprop_tools_test(float*);
#ifdef __cplusplus
extern "C"
#endif
void backprop_tools_control(float* state, float* actions);
#ifdef __cplusplus
extern "C"
#endif
char* backprop_tools_get_checkpoint_name();


