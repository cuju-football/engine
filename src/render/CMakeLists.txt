# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

# Every source under src/render belongs to the library. The flags are the CPU backend's, which
# contract nothing, so each fused multiply-add is an explicit `std::fma`.
file(GLOB_RECURSE cuju_render_sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
if(NOT cuju_render_sources)
    return()
endif()
find_package(Threads REQUIRED)
add_library(cuju_render SHARED ${cuju_render_sources})
separate_arguments(cuju_render_flags UNIX_COMMAND "${CUJU_CPU_FLAGS}")
set_source_files_properties(${cuju_render_sources} PROPERTIES
    COMPILE_OPTIONS "${cuju_render_flags};${cuju_no_contraction}")
target_include_directories(cuju_render PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cuju_render PRIVATE cuju_exact_flags Threads::Threads)
if(NOT MSVC)
    target_compile_options(cuju_render PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
endif()
if(WIN32)
    file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/render_api.h cuju_render_lines
        REGEX "^[A-Za-z][^(]*[ *]cuju_render_[a-z0-9_]+[(]")
    set(cuju_render_exports "")
    foreach(line IN LISTS cuju_render_lines)
        if(line MATCHES "^[A-Za-z][^(]*[ *](cuju_render_[a-z0-9_]+)[(]")
            list(APPEND cuju_render_exports ${CMAKE_MATCH_1})
        endif()
    endforeach()
    list(REMOVE_DUPLICATES cuju_render_exports)
    list(SORT cuju_render_exports)
    list(JOIN cuju_render_exports "\n    " cuju_render_exports)
    file(CONFIGURE OUTPUT ${PROJECT_BINARY_DIR}/cuju_render.def
        CONTENT "EXPORTS\n    ${cuju_render_exports}\n" @ONLY)
    target_sources(cuju_render PRIVATE ${PROJECT_BINARY_DIR}/cuju_render.def)
endif()
if(SKBUILD)
    install(TARGETS cuju_render LIBRARY DESTINATION cuju RUNTIME DESTINATION cuju
        ARCHIVE DESTINATION cuju EXCLUDE_FROM_ALL)
endif()
