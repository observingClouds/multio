# Optional chunk compression (zarr v3 codec zstd) with libzstd, plain C API (zstd >= 1.4).
# Found with ZSTD_ROOT / CMAKE_PREFIX_PATH / pkg-config; MULTIO_ZARR_ZSTD=OFF builds without it
# (the action then rejects `compression: {codec: zstd}` at configuration time).
option( MULTIO_ZARR_ZSTD "zarr action: zstd compression if libzstd is found" ON )

set( multio_zarr_zstd_libs )
set( multio_zarr_zstd_includes )
set( multio_zarr_zstd_definitions )
if( MULTIO_ZARR_ZSTD )
    find_package( PkgConfig QUIET )
    if( PKG_CONFIG_FOUND )
        pkg_check_modules( PC_ZSTD QUIET libzstd )
    endif()
    find_path( ZSTD_INCLUDE_DIR zstd.h
        HINTS ${ZSTD_ROOT} $ENV{ZSTD_ROOT} ${ZSTD_DIR} $ENV{ZSTD_DIR} ${PC_ZSTD_INCLUDEDIR} ${PC_ZSTD_INCLUDE_DIRS}
        PATH_SUFFIXES include )
    find_library( ZSTD_LIBRARY NAMES zstd libzstd
        HINTS ${ZSTD_ROOT} $ENV{ZSTD_ROOT} ${ZSTD_DIR} $ENV{ZSTD_DIR} ${PC_ZSTD_LIBDIR} ${PC_ZSTD_LIBRARY_DIRS}
        PATH_SUFFIXES lib lib64 )
    if( ZSTD_INCLUDE_DIR AND ZSTD_LIBRARY )
        ecbuild_info( "zarr action: zstd compression with ${ZSTD_LIBRARY} (${ZSTD_INCLUDE_DIR})" )
        set( multio_zarr_zstd_libs ${ZSTD_LIBRARY} )
        set( multio_zarr_zstd_includes ${ZSTD_INCLUDE_DIR} )
        set( multio_zarr_zstd_definitions MULTIO_ZARR_HAVE_ZSTD )
    else()
        ecbuild_warn( "zarr action: libzstd not found (set ZSTD_ROOT), building without zstd compression" )
    endif()
endif()
set( MULTIO_ZARR_HAVE_ZSTD ${multio_zarr_zstd_definitions} CACHE INTERNAL "zarr action built with zstd" )

find_package( Threads REQUIRED )

ecbuild_add_library(

    TARGET multio-action-zarr

    TYPE SHARED # Due to reliance on factory self registration this library cannot be static

    SOURCES
        Zarr.cc
        Zarr.h
        ZarrCodec.cc
        ZarrCodec.h
        ZarrStore.cc
        ZarrStore.h
        ZarrUtil.cc
        ZarrUtil.h

    PRIVATE_INCLUDES
        ${ECKIT_INCLUDE_DIRS}
        ${multio_zarr_zstd_includes}

    PRIVATE_DEFINITIONS
        ${multio_zarr_zstd_definitions}

    CONDITION

    PUBLIC_LIBS
        multio

    PRIVATE_LIBS
        ${multio_zarr_zstd_libs}
        Threads::Threads
)
