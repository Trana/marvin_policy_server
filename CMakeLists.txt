cmake_minimum_required(VERSION 3.16)
project(marvin_policy_server LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
ament_python_install_package(${PROJECT_NAME})

# Both frontends use the same Python package. The native frontend starts the
# installed helper through an inherited private socket, never a network listener.
function(policy_entrypoint name module)
  set(script "${CMAKE_CURRENT_BINARY_DIR}/entrypoints/${name}")
  file(GENERATE OUTPUT "${script}" CONTENT
    "#!${Python3_EXECUTABLE}\nfrom marvin_policy_server.${module} import main\nif __name__ == '__main__':\n    main()\n")
  install(PROGRAMS "${script}" DESTINATION lib/${PROJECT_NAME})
endfunction()
policy_entrypoint(marvin_policy_server marvin_policy_server)
policy_entrypoint(marvin_policy_server_native native_server)
policy_entrypoint(marvin_delay_test delay_test_node)
policy_entrypoint(joy_command_recorder joy_command_recorder_node)
policy_entrypoint(joy_command_replay joy_command_replay_node)

install(DIRECTORY launch policy DESTINATION share/${PROJECT_NAME})
install(DIRECTORY docs DESTINATION share/${PROJECT_NAME})
option(BUILD_ROS_IO "Build the opt-in C++ policy frontend" ON)
enable_testing()
add_subdirectory(native_io)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  set(POLICY_TEST_PYTHON "${Python3_EXECUTABLE}" CACHE FILEPATH "Python with policy/PyTorch test dependencies")
  ament_add_pytest_test(policy_functional test
    PYTHON_EXECUTABLE "${POLICY_TEST_PYTHON}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    ENV "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1"
        "PYTEST_ADDOPTS=--ignore=test/test_copyright.py --ignore=test/test_flake8.py --ignore=test/test_pep257.py"
    TIMEOUT 180)
endif()
ament_package()
