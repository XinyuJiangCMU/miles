#include <hip/hip_runtime.h>
#include <cstdio>
#include <unistd.h>
#include <sys/wait.h>
#define CK(x) do{hipError_t e=(x); if(e!=hipSuccess){printf("ERR %s:%s\n",#x,hipGetErrorString(e));fflush(stdout);_exit(2);}}while(0)
int main(){
  int p2c[2], c2p[2]; if(pipe(p2c)||pipe(c2p))return 1;
  pid_t pid=fork();
  if(pid==0){ // child consumer — own HIP init
    close(p2c[1]); close(c2p[0]); CK(hipSetDevice(0)); CK(hipFree(0));
    for(int i=0;i<16;i++){ hipIpcMemHandle_t h;
      if(read(p2c[0],&h,sizeof(h))!=(ssize_t)sizeof(h))break;
      void* ptr=nullptr; CK(hipIpcOpenMemHandle(&ptr,h,hipIpcMemLazyEnablePeerAccess));
      CK(hipIpcCloseMemHandle(ptr)); char a=1; if(write(c2p[1],&a,1)){} }
    _exit(0); }
  // parent producer — own HIP init
  close(p2c[0]); close(c2p[1]); CK(hipSetDevice(0)); CK(hipFree(0));
  size_t f0,t0; CK(hipMemGetInfo(&f0,&t0)); double first=0,last=0; size_t SZ=256ull*1024*1024;
  for(int i=0;i<16;i++){ void* p=nullptr; CK(hipExtMallocWithFlags(&p,SZ,hipDeviceMallocDefault));
    hipIpcMemHandle_t h; CK(hipIpcGetMemHandle(&h,p)); if(write(p2c[1],&h,sizeof(h))){}
    char a; if(read(c2p[0],&a,1)!=1)break; CK(hipFree(p));
    size_t fr,tt; CK(hipMemGetInfo(&fr,&tt)); double g=fr/1073741824.0;
    if(i==1)first=g; if(i==15)last=g; }
  close(p2c[1]); wait(NULL);
  printf("RESULT churn2p: drop=%.3f GB (%.1f MB/iter)\n",first-last,(first-last)*1024/14); fflush(stdout);
  return 0; }
