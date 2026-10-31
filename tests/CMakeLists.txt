# Each executable target matches its source filename without the .cpp suffix.
add_executable(build_segment_index build_segment_index.cpp)
add_executable(build_quadtree_index build_quadtree_index.cpp)
add_executable(search_zorder search_zorder.cpp)
add_executable(search_quadtree search_quadtree.cpp)
add_executable(search_zorder_label search_zorder_label.cpp)
add_executable(search_label search_label.cpp)
add_executable(search_mixed search_mixed.cpp)

foreach(target IN ITEMS build_segment_index build_quadtree_index search_zorder
        search_quadtree search_zorder_label search_label search_mixed)
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
            --bin-dir "$<TARGET_FILE_DIR:build_segment_index>"
            --work-dir "${CMAKE_CURRENT_BINARY_DIR}/cli-smoke-fixtures")
endif()
