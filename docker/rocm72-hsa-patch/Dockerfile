ARG MILES_IMAGE=xinyujiangcmu/miles:rocm720-mi35x-20260621
FROM rocm/dev-ubuntu-22.04:7.2-complete AS rocr
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake xxd libelf-dev libnuma-dev libzstd-dev libdrm-dev libxml2-dev pkg-config
RUN apt-get install -y rocm-llvm-dev7.2.0 || true
# TODO: remove this patch once the fix ships in a ROCm 7.2.x release (or use ROCm 7.0/7.1).
COPY amdgpu_bo_free.patch /tmp/amdgpu_bo_free.patch
RUN git clone --depth 1 -b rocm-7.2.0 https://github.com/ROCm/ROCR-Runtime /src \
    && cd /src && git apply /tmp/amdgpu_bo_free.patch \
    && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm -DROCM_PATCH_VERSION=70200 \
    && cmake --build build --target hsa-runtime64 -j"$(nproc)" \
    && cp "$(find build -name 'libhsa-runtime64.so.1.18.*' -type f | head -1)" /opt/libhsa.patched
FROM ${MILES_IMAGE}
COPY --from=rocr /opt/libhsa.patched /opt/rocm/lib/libhsa-runtime64.so.1.18.70200
