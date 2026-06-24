# Fixes for the ROCm 7.2 hipFree-after-IPC-export page leak. Each releases a transient libdrm
# import reference (amdgpu_bo_import) that 7.2 takes to manage IPC-handle dedup metadata but
# never frees on the leaking paths, so the kernel cannot reclaim the exporter pages.
f = "runtime/hsa-runtime/core/runtime/runtime.cpp"
s = open(f).read()

# Patch 1 (exporter): in Runtime::FreeMemory, after the existing free-time metadata-clear, add
# the missing amdgpu_bo_free of the import reference IPCCreate stored in ldrm_bo. Keeping the
# set_metadata(zero) ensures a reused GEM does not carry a stale IPC handle.
e_old = '''        DRM_CALL(amdgpu_bo_set_metadata(it->second.ldrm_bo, &zero_metadata));
      }'''
e_new = '''        DRM_CALL(amdgpu_bo_set_metadata(it->second.ldrm_bo, &zero_metadata));

        // Exporter-side fix (ROCm 7.2): release the import reference IPCCreate took to write
        // the dedup metadata. The set_metadata(zero) above is preserved (free-time clear);
        // this only adds the missing free so the kernel reclaims the pages after hipFree.
        DRM_CALL(amdgpu_bo_free(it->second.ldrm_bo));
        it->second.ldrm_bo = nullptr;
      }'''
assert s.count(e_old) == 1, ("exporter", s.count(e_old))
s = s.replace(e_old, e_new)

# Patch 2 (consumer): in Runtime::IPCAttach, release the import reference right after it is used
# to read the dedup metadata (info is already populated), before the validation return.
# Otherwise mapMemoryToNodes/fixFragment overwrites this map slot and orphans the BO, so
# IPCDetach never frees it -- the leak that only manifests when a consumer maps the handle.
c_old = '''          int ret = DRM_CALL(amdgpu_bo_query_info(allocation_map_[importAddress].ldrm_bo, &info));'''
c_new = '''          int ret = DRM_CALL(amdgpu_bo_query_info(allocation_map_[importAddress].ldrm_bo, &info));

          // Consumer-side fix (ROCm 7.2): the import reference is only needed to read the
          // dedup metadata above (now in `info`); the GPU mapping is owned by the thunk.
          // Release it here -- before the validation return below, so the error path frees it
          // too -- otherwise mapMemoryToNodes/fixFragment orphans it and IPCDetach never frees it.
          DRM_CALL(amdgpu_bo_free(allocation_map_[importAddress].ldrm_bo));
          allocation_map_[importAddress].ldrm_bo = NULL;'''
assert s.count(c_old) == 1, ("consumer", s.count(c_old))
s = s.replace(c_old, c_new)

# Patch 3 (pre-existing dedup-hit leak, independent of the two above): when the buffer already
# carries IPC metadata (a re-export of a still-live buffer), IPCCreate's if-branch reads the
# existing handle[7] but neither stores nor frees this transient import, leaking it. The first
# export's reference is already tracked in allocation_map_ and freed by FreeMemory, so release
# this redundant one. Not hit by the base colocate path (which exports a fresh buffer each step
# and takes the else-branch); closes a reviewer-flagged edge.
d_old = '''      if (!DRM_CALL(amdgpu_bo_query_info(res.buf_handle, &info)) && !!info.metadata.size_metadata) {
        handle->handle[7] = info.metadata.umd_metadata[0];
      } else {'''
d_new = '''      if (!DRM_CALL(amdgpu_bo_query_info(res.buf_handle, &info)) && !!info.metadata.size_metadata) {
        handle->handle[7] = info.metadata.umd_metadata[0];
        // Pre-existing dedup-hit leak fix: release this redundant re-export import reference.
        DRM_CALL(amdgpu_bo_free(res.buf_handle));
      } else {'''
assert s.count(d_old) == 1, ("dedup", s.count(d_old))
s = s.replace(d_old, d_new)

open(f, "w").write(s)
print("applied 3 fixes (FreeMemory exporter + IPCAttach consumer + IPCCreate dedup-hit)")
