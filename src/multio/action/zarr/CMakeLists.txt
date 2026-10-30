ecbuild_add_library(

    TARGET multio-action-zarr

    TYPE SHARED # Due to reliance on factory self registration this library cannot be static

    SOURCES
        Zarr.cc
        Zarr.h
        ZarrStore.cc
        ZarrStore.h
        ZarrUtil.cc
        ZarrUtil.h

    PRIVATE_INCLUDES
        ${ECKIT_INCLUDE_DIRS}

    CONDITION

    PUBLIC_LIBS
        multio
)
