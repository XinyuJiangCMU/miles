ARG MILES_IMAGE=xinyujiangcmu/miles:rocm720-mi35x-20260621
FROM rocm/dev-ubuntu-22.04:7.2-complete AS rocr
ENV DEBIAN_FRONTEND=noninteractive ROCM_PATCH_VERSION=70200
RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake xxd libelf-dev libnuma-dev libzstd-dev libdrm-dev libxml2-dev pkg-config
# Unpacking rocm-llvm-dev conflicts with the image's rocm-llvm at configure time, but it still
# lays down the LLVM CMake configs + static archives the ROCR build's find_package needs.
RUN apt-get install -y rocm-llvm-dev7.2.0 || true
RUN git clone --depth 1 -b rocm-7.2.0 https://github.com/ROCm/ROCR-Runtime /src
# Two-part fix for the ROCm 7.2 hipFree-after-IPC-export page leak. Both release a transient
# libdrm import reference (amdgpu_bo_import) that 7.2 takes but never frees.
COPY apply_fix.py /tmp/apply_fix.py
RUN cd /src && python3 /tmp/apply_fix.py
RUN cd /src && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm -DROCM_PATCH_VERSION=70200 \
    && cmake --build build --target hsa-runtime64 -j"$(nproc)" \
    && cp "$(find build -name 'libhsa-runtime64.so.1.18.*' -type f | head -1)" /opt/libhsa.patched
FROM ${MILES_IMAGE}
COPY --from=rocr /opt/libhsa.patched /opt/rocm/lib/libhsa-runtime64.so.1.18.70200
