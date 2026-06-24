// Minimal repro: does hipFree return pages of an IPC-exported block?
// build: hipcc churn_ipc.cpp -o churn_ipc
// arg1: "export" (call hipIpcGetMemHandle) or "control" (do not)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>

#define CK(x) do{ hipError_t e=(x); if(e!=hipSuccess){printf("ERR %s: %s\n", #x, hipGetErrorString(e)); return 1;} }while(0)

int main(int argc, char** argv){
    bool doExport = (argc>1 && strcmp(argv[1],"export")==0);
    const size_t SZ = (size_t)256*1024*1024; // 256MB
    CK(hipFree(0)); // init ctx
    size_t f0,t0; CK(hipMemGetInfo(&f0,&t0));
    printf("mode=%s  size=256MB  free0=%.2f GB\n", doExport?"EXPORT":"CONTROL", f0/1073741824.0);
    double first=0, last=0;
    for(int i=0;i<16;i++){
        void* p=nullptr;
        CK(hipExtMallocWithFlags(&p, SZ, hipDeviceMallocDefault));
        if(doExport){
            hipIpcMemHandle_t h;
            CK(hipIpcGetMemHandle(&h, p));   // <-- the trigger
        }
        CK(hipFree(p));
        size_t fr,tt; CK(hipMemGetInfo(&fr,&tt));
        double freeGB=fr/1073741824.0;
        printf("  iter %2d: free=%.3f GB\n", i, freeGB);
        if(i==1) first=freeGB;
        if(i==15) last=freeGB;
    }
    printf("RESULT mode=%s  free drop iter1->iter15 = %.3f GB  (%.1f MB/iter)\n",
           doExport?"EXPORT":"CONTROL", first-last, (first-last)*1024.0/14.0);
    return 0;
}
