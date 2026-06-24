# Overlay the ROCm 7.2 hipFree-IPC-export memory-leak fix on the miles image.
#
#   docker build -t <your-tag> -f docker/rocm72-hsa-patch/Dockerfile docker/rocm72-hsa-patch
#
# ROCm 7.2.0-7.2.4 regressed hipFree so it no longer returns the physical pages of an
# allocation once it has been IPC-exported. This swaps libhsa-runtime64.so for a 7.2.0 build
# that includes the missing amdgpu_bo_free() in Runtime::FreeMemory (see amdgpu_bo_free.patch).
# The SONAME (libhsa-runtime64.so.1) is unchanged, so nothing else needs rebuilding.
ARG MILES_IMAGE=xinyujiangcmu/miles:rocm720-mi35x-20260621
FROM ${MILES_IMAGE}
COPY libhsa-runtime64.so.1.18.70200 /opt/rocm/lib/libhsa-runtime64.so.1.18.70200
