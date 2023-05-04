set(FILE_SRC
        debug.cpp
        auxiliary.cpp
        subgraph_enumeration.cpp
        subgraph_retrieval.cpp
)

# Add source files for the executable
add_executable(debug.o ${FILE_SRC})

# Link against the libraries
# target_link_libraries(debug.o configuration)
target_link_libraries(debug.o graph)
target_link_libraries(debug.o index)
target_link_libraries(debug.o nd)
target_link_libraries(debug.o utility)
target_link_libraries(debug.o rt)

# Set the output directory for built binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
