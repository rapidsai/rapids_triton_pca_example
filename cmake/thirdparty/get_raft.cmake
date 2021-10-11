#=============================================================================
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

function(find_and_configure_raft)

    set(oneValueArgs VERSION FORK PINNED_TAG)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN} )

    if(DEFINED CPM_raft_SOURCE OR NOT DISABLE_FORCE_CLONE_RAFT)
      set(CPM_DL_ALL_CACHE ${CPM_DOWNLOAD_ALL})
      set(CPM_DOWNLOAD_ALL ON)
    endif()

    rapids_cpm_find(raft ${PKG_VERSION}
      GLOBAL_TARGETS      raft::raft
      BUILD_EXPORT_SET    ${BACKEND_TARGET}-exports
      INSTALL_EXPORT_SET  ${BACKEND_TARGET}-exports
        CPM_ARGS
            GIT_REPOSITORY https://github.com/${PKG_FORK}/raft.git
            GIT_TAG        ${PKG_PINNED_TAG}
            SOURCE_SUBDIR  cpp
            OPTIONS
              "BUILD_TESTS OFF"
    )

    if(raft_ADDED)
      message(VERBOSE "RAPIDS_TRITON_BACKEND: Using RAFT located in ${raft_SOURCE_DIR}")
    else()
      message(VERBOSE "RAPIDS_TRITON_BACKEND: Using RAFT located in ${raft_DIR}")
    endif()

    if(DEFINED CPM_raft_SOURCE OR NOT DISABLE_FORCE_CLONE_RAFT)
      set(CPM_DOWNLOAD_ALL ${CPM_DL_ALL_CACHE})
    endif()

endfunction()

set(RAFT_MIN_VERSION "21.10.00")
set(RAFT_BRANCH_VERSION "21.10")

# Change pinned tag here to test a commit in CI
# To use a different RAFT locally, set the CMake variable
# CPM_raft_SOURCE=/path/to/local/raft
find_and_configure_raft(VERSION    ${RAFT_MIN_VERSION}
                        FORK       rapidsai
                        PINNED_TAG branch-${RAFT_BRANCH_VERSION}
                        )
