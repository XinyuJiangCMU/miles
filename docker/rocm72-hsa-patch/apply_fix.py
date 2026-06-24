f = "runtime/hsa-runtime/core/runtime/runtime.cpp"
s = open(f).read()

edits = [
    ("exporter",
     '''        DRM_CALL(amdgpu_bo_set_metadata(it->second.ldrm_bo, &zero_metadata));
      }''',
     '''        DRM_CALL(amdgpu_bo_set_metadata(it->second.ldrm_bo, &zero_metadata));
        DRM_CALL(amdgpu_bo_free(it->second.ldrm_bo));
        it->second.ldrm_bo = nullptr;
      }'''),
    ("consumer",
     '''          int ret = DRM_CALL(amdgpu_bo_query_info(allocation_map_[importAddress].ldrm_bo, &info));''',
     '''          int ret = DRM_CALL(amdgpu_bo_query_info(allocation_map_[importAddress].ldrm_bo, &info));
          DRM_CALL(amdgpu_bo_free(allocation_map_[importAddress].ldrm_bo));
          allocation_map_[importAddress].ldrm_bo = NULL;'''),
    ("dedup",
     '''      if (!DRM_CALL(amdgpu_bo_query_info(res.buf_handle, &info)) && !!info.metadata.size_metadata) {
        handle->handle[7] = info.metadata.umd_metadata[0];
      } else {''',
     '''      if (!DRM_CALL(amdgpu_bo_query_info(res.buf_handle, &info)) && !!info.metadata.size_metadata) {
        handle->handle[7] = info.metadata.umd_metadata[0];
        DRM_CALL(amdgpu_bo_free(res.buf_handle));
      } else {'''),
]
for tag, old, new in edits:
    assert s.count(old) == 1, (tag, s.count(old))
    s = s.replace(old, new)
open(f, "w").write(s)
