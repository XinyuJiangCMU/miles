# Two-part fix for the ROCm 7.2 hipFree-after-IPC-export page leak. Both release a transient
# libdrm import reference (amdgpu_bo_import) that 7.2 takes to manage IPC-handle dedup metadata
# but never frees on the leaking paths, so the kernel cannot reclaim the exporter pages.
f = "runtime/hsa-runtime/core/runtime/runtime.cpp"
s = open(f).read()

# Patch 1 (exporter): in Runtime::FreeMemory, after the existing free-time metadata-clear,
# add the missing amdgpu_bo_free of the import reference IPCCreate stored in ldrm_bo. This
# keeps the original set_metadata(zero) (so a reused GEM does not carry a stale IPC handle).
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

# Patch 2 (consumer): in Runtime::IPCAttach, release the import reference right after it is
# used to read the dedup metadata (info is already populated), before the validation return.
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

open(f, "w").write(s)
print("dual-fix v2 applied (FreeMemory exporter + IPCAttach consumer-before-validation)")
