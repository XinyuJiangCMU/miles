ARG MILES_IMAGE=xinyujiangcmu/miles:rocm720-mi35x-20260621
FROM rocm/dev-ubuntu-22.04:7.2-complete AS rocr
ENV DEBIAN_FRONTEND=noninteractive ROCM_PATCH_VERSION=70200
RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake xxd libelf-dev libnuma-dev libzstd-dev libdrm-dev libxml2-dev pkg-config
# rocm-llvm-dev conflicts with the image's rocm-llvm at configure time, but unpacking it
# (|| true) lays down the LLVM CMake configs + static archives the ROCR build's
# find_package(Clang/LLVM) needs.
RUN apt-get install -y rocm-llvm-dev7.2.0 || true
RUN git clone --depth 1 -b rocm-7.2.0 https://github.com/ROCm/ROCR-Runtime /src
# Apply the fix in Runtime::IPCCreate: release the transient libdrm import reference
# (covers both metadata branches), so hipFree returns the pages of IPC-exported allocs.
RUN cd /src && python3 - <<'PY'
f="runtime/hsa-runtime/core/runtime/runtime.cpp"
s=open(f).read()
old="""        DRM_CALL(amdgpu_bo_set_metadata(res.buf_handle, &buf_info));
        allocation_map_[ptr].ldrm_bo = res.buf_handle;
      }"""
new="""        DRM_CALL(amdgpu_bo_set_metadata(res.buf_handle, &buf_info));
      }

      // Release the transient libdrm import reference (regression: keeping it prevented
      // the kernel from reclaiming pages of IPC-exported allocations after hipFree).
      DRM_CALL(amdgpu_bo_free(res.buf_handle));"""
assert s.count(old)==1, s.count(old)
open(f,"w").write(s.replace(old,new)); print("patched")
PY
RUN cd /src && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm \
    && cmake --build build --target hsa-runtime64 -j"$(nproc)" \
    && cp "$(find build -name 'libhsa-runtime64.so.1.18.*' -type f | head -1)" /opt/libhsa.patched
FROM ${MILES_IMAGE}
COPY --from=rocr /opt/libhsa.patched /opt/rocm/lib/libhsa-runtime64.so.1.18.70200
