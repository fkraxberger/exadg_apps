cmake_minimum_required(VERSION 3.30)

set(CMAKE_CXX_STANDARD 17)

project(ExaDG_Apps)

find_package(deal.II 9.5 QUIET REQUIRED HINTS ${DEAL_II_DIR})
find_package(EXADG REQUIRED HINTS ${EXADG_DIR})

add_executable(valley ${CMAKE_CURRENT_LIST_DIR}/applications/acoustic_conservation_laws/valley/solver.cpp)
deal_ii_setup_target(valley DEBUG)
target_link_libraries(valley EXADG::exadg)
