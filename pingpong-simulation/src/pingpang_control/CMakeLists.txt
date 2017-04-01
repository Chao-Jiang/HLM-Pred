##catkin_make --force-cmake -G"Eclipse CDT4 - Unix Makefiles"  -DCMAKE_BUILD_TYPE=Debug
##catkin_make install -DCMAKE_INSTALL_PREFIX=devel/

cmake_minimum_required(VERSION 2.8.3)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") 
project(pingpang_control)


find_package(catkin REQUIRED COMPONENTS
  message_generation
  rospy
  genmsg 
  actionlib_msgs 
  actionlib
  cv_bridge
  image_transport
)


find_package(Boost REQUIRED COMPONENTS system)


catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################


add_message_files(
   FILES
 )


add_service_files(
   FILES
   Table_Tennis.srv
 )


add_action_files(
   FILES
 )


generate_messages(
   DEPENDENCIES
   std_msgs  
   actionlib_msgs
 )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################


## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES basic
  CATKIN_DEPENDS message_generation rospy
  DEPENDS system_lib
)

###########
## Build ##
###########


include_directories(include)

include_directories(
   /usr/include/python2.7
  ${catkin_INCLUDE_DIRS}

)

LINK_DIRECTORIES(
#  /home/yangqingyan//work/lib
#  /home/yangqingyan/opencv-2.4.9/install/lib
#  ${PCL_LIBRARY_DIRS}
 )

#add_library(
##   example
##   src/example_wrap.cxx src/example.cpp 
# )


# add_executable(test_node src/test.cpp)
#  target_link_libraries(test_node
#    ${catkin_LIBRARIES}
#    -pthread nsl X11 caffe glog gflags 
#    protobuf boost_system  boost_filesystem 
#    boost_thread boost_coroutine boost_context boost_regex
#    m hdf5_hl hdf5 leveldb snappy lmdb
#    opencv_core opencv_highgui opencv_imgproc opencv_objdetect 
#    stdc++ cblas atlas 
#  )

#############
## Install ##
#############

 install(PROGRAMS
   # src/py_pkg/Functions.py
   # src/py_pkg/Camera.py
   # src/py_pkg/car_control.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )

## Mark executables and/or libraries for installation
##install(TARGETS foo foo
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

