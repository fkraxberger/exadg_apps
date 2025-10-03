cmake_minimum_required(VERSION 3.24)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

project(ExaDG_Apps)

find_package(deal.II 9.5 QUIET REQUIRED HINTS $ENV{DEAL_II_DIR})
find_package(EXADG REQUIRED HINTS $ENV{EXADG_DIR})

# add_executable(valley ${CMAKE_CURRENT_LIST_DIR}/applications/acoustic_conservation_laws/valley/solver.cpp)
# deal_ii_setup_target(valley DEBUG)
# target_link_libraries(valley EXADG::exadg)

add_executable(pml
        ${CMAKE_CURRENT_LIST_DIR}/applications/acoustic_conservation_laws/pml/solver.cpp
)
deal_ii_setup_target(pml DEBUG)
target_link_libraries(pml EXADG::exadg)

# add_executable(cube_pml
#         ${CMAKE_CURRENT_LIST_DIR}/applications/acoustic_conservation_laws/cube_pml/solver.cpp
# )
# deal_ii_setup_target(cube_pml DEBUG)
# target_link_libraries(cube_pml EXADG::exadg)

add_executable(duct_pml
        ${CMAKE_CURRENT_LIST_DIR}/applications/acoustic_conservation_laws/duct_pml/solver.cpp
)
deal_ii_setup_target(duct_pml DEBUG)
target_link_libraries(duct_pml EXADG::exadg)

# add_executable(cube_pml_no_overlap
#         ${CMAKE_CURRENT_LIST_DIR}/applications/acoustic_conservation_laws/cube_pml_no_overlap/solver.cpp
# )
# deal_ii_setup_target(cube_pml_no_overlap DEBUG)
# target_link_libraries(cube_pml_no_overlap EXADG::exadg)

# add_executable(cube_pml_with_overlap
#         ${CMAKE_CURRENT_LIST_DIR}/applications/acoustic_conservation_laws/cube_pml_with_overlap/solver.cpp
# )
# deal_ii_setup_target(cube_pml_with_overlap DEBUG)
# target_link_libraries(cube_pml_with_overlap EXADG::exadg)

add_executable(cube_pml_opposing
        ${CMAKE_CURRENT_LIST_DIR}/applications/acoustic_conservation_laws/cube_pml_opposing/solver.cpp
)
deal_ii_setup_target(cube_pml_opposing DEBUG)
target_link_libraries(cube_pml_opposing EXADG::exadg)

add_executable(valley_sourceRegion_w300
        ${CMAKE_CURRENT_LIST_DIR}/applications/acoustic_conservation_laws/valley_sourceRegion_w300/solver.cpp
)
deal_ii_setup_target(valley_sourceRegion_w300 DEBUG)
target_link_libraries(valley_sourceRegion_w300 EXADG::exadg)

add_executable(valley_sourceRegion_w450
        ${CMAKE_CURRENT_LIST_DIR}/applications/acoustic_conservation_laws/valley_sourceRegion_w450/solver.cpp
)
deal_ii_setup_target(valley_sourceRegion_w450 DEBUG)
target_link_libraries(valley_sourceRegion_w450 EXADG::exadg)

add_executable(valley_sourceRegion_w600
        ${CMAKE_CURRENT_LIST_DIR}/applications/acoustic_conservation_laws/valley_sourceRegion_w600/solver.cpp
)
deal_ii_setup_target(valley_sourceRegion_w600 DEBUG)
target_link_libraries(valley_sourceRegion_w600 EXADG::exadg)

# add_executable(valley_sourceRegion_w300_impulseExc
#         ${CMAKE_CURRENT_LIST_DIR}/applications/acoustic_conservation_laws/valley_sourceRegion_w300_impulseExc/solver.cpp
# )
# deal_ii_setup_target(valley_sourceRegion_w300_impulseExc DEBUG)
# target_link_libraries(valley_sourceRegion_w300_impulseExc EXADG::exadg)

add_executable(simple_LSW
        ${CMAKE_CURRENT_LIST_DIR}/applications/acoustic_conservation_laws/simple_LSW/solver.cpp
)
deal_ii_setup_target(simple_LSW DEBUG)
target_link_libraries(simple_LSW EXADG::exadg)

add_executable(room
        ${CMAKE_CURRENT_LIST_DIR}/applications/acoustic_conservation_laws/room/solver.cpp
)
deal_ii_setup_target(room DEBUG)
target_link_libraries(room EXADG::exadg)