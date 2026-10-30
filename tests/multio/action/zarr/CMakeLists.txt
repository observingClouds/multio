
set( PREFIX multio_tests_action_zarr )

ecbuild_add_test(
    TARGET ${PREFIX}
    SOURCES TestZarr.cc
    NO_AS_NEEDED
    LIBS multio-action-select multio-action-aggregate multio-action-zarr
)
