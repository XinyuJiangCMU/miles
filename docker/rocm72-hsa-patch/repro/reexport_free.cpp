// F validation: each iter alloc -> export 3x (1st else-branch stores ldrm_bo; 2nd/3rd dedup-hit
// if-branch) -> free. Without patch 3 the 2 if-branch imports are never released and pin the
// GEM so hipFree cannot return the pages -> leak. With patch 3 -> flat.
#include <hip/hip_runtime.h>
#include <cstdio>
#define CK(x) do{hipError_t e=(x); if(e!=hipSuccess){printf("ERR %s:%s\n",#x,hipGetErrorString(e));return 2;}}while(0)
int main(){
  CK(hipFree(0));
  size_t SZ=256ull*1024*1024; size_t f0,t0; CK(hipMemGetInfo(&f0,&t0)); double first=0,last=0;
  for(int i=0;i<16;i++){
    void* p=nullptr; CK(hipExtMallocWithFlags(&p,SZ,hipDeviceMallocDefault));
    hipIpcMemHandle_t h;
    CK(hipIpcGetMemHandle(&h,p));   // 1st: else-branch
    CK(hipIpcGetMemHandle(&h,p));   // 2nd: dedup-hit if-branch
    CK(hipIpcGetMemHandle(&h,p));   // 3rd: dedup-hit if-branch
    CK(hipFree(p));
    size_t fr,tt; CK(hipMemGetInfo(&fr,&tt)); double g=fr/1073741824.0;
    if(i==1)first=g; if(i==15)last=g;
  }
  printf("RESULT reexport_free: drop iter1->15 = %.3f GB (%.1f MB/iter)\n",first-last,(first-last)*1024/14);
  return 0;
}
