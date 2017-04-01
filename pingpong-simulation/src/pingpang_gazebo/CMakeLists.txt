##catkin_make --force-cmake -G"Eclipse CDT4 - Unix Makefiles"  -DCMAKE_BUILD_TYPE=Debug
##catkin_make install -DCMAKE_INSTALL_PREFIX=devel/

cmake_minimum_required(VERSION 2.8.3)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") 
project(pingpang_gazebo)


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
 )


#############
## Install ##
#############

 install(PROGRAMS
   src/script/Controller.py
   src/script/position.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )


