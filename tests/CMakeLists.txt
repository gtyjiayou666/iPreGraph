# Preserve the public executable names while using descriptive source filenames.
add_executable(buildindex build_segment_index.cpp)
add_executable(buildindex-quadtree build_quadtree_index.cpp)
add_executable(search_multi-zorder search_zorder.cpp)
add_executable(search_multi-quadtree search_quadtree.cpp)
add_executable(search_multi-zorder-label search_zorder_label.cpp)
add_executable(search_multi-or-json search_label.cpp)
add_executable(search_mixed search_mixed.cpp)

foreach(target IN ITEMS buildindex buildindex-quadtree search_multi-zorder
        search_multi-quadtree search_multi-zorder-label search_multi-or-json search_mixed)
    target_link_libraries(${target} PRIVATE ipregraph_options)
endforeach()

if(BUILD_TESTING)
    add_executable(irange_multi_test irange_multi_test.cpp)
    target_link_libraries(irange_multi_test PRIVATE ipregraph_options)
    add_test(NAME irange_multi
        COMMAND irange_multi_test "${CMAKE_CURRENT_BINARY_DIR}/irange-fixtures")

    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME cli_smoke
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/cli_smoke_test.py"
            --bin-dir "$<TARGET_FILE_DIR:buildindex>"
            --work-dir "${CMAKE_CURRENT_BINARY_DIR}/cli-smoke-fixtures")
endif()
