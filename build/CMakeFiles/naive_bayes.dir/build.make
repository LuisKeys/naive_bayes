# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lucho/Projects/SelfDrivingCar/naive_bayes

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lucho/Projects/SelfDrivingCar/naive_bayes/build

# Include any dependencies generated for this target.
include CMakeFiles/naive_bayes.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/naive_bayes.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/naive_bayes.dir/flags.make

CMakeFiles/naive_bayes.dir/src/main.cpp.o: CMakeFiles/naive_bayes.dir/flags.make
CMakeFiles/naive_bayes.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lucho/Projects/SelfDrivingCar/naive_bayes/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/naive_bayes.dir/src/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/naive_bayes.dir/src/main.cpp.o -c /home/lucho/Projects/SelfDrivingCar/naive_bayes/src/main.cpp

CMakeFiles/naive_bayes.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/naive_bayes.dir/src/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lucho/Projects/SelfDrivingCar/naive_bayes/src/main.cpp > CMakeFiles/naive_bayes.dir/src/main.cpp.i

CMakeFiles/naive_bayes.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/naive_bayes.dir/src/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lucho/Projects/SelfDrivingCar/naive_bayes/src/main.cpp -o CMakeFiles/naive_bayes.dir/src/main.cpp.s

CMakeFiles/naive_bayes.dir/src/main.cpp.o.requires:

.PHONY : CMakeFiles/naive_bayes.dir/src/main.cpp.o.requires

CMakeFiles/naive_bayes.dir/src/main.cpp.o.provides: CMakeFiles/naive_bayes.dir/src/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/naive_bayes.dir/build.make CMakeFiles/naive_bayes.dir/src/main.cpp.o.provides.build
.PHONY : CMakeFiles/naive_bayes.dir/src/main.cpp.o.provides

CMakeFiles/naive_bayes.dir/src/main.cpp.o.provides.build: CMakeFiles/naive_bayes.dir/src/main.cpp.o


CMakeFiles/naive_bayes.dir/src/classifier.cpp.o: CMakeFiles/naive_bayes.dir/flags.make
CMakeFiles/naive_bayes.dir/src/classifier.cpp.o: ../src/classifier.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lucho/Projects/SelfDrivingCar/naive_bayes/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/naive_bayes.dir/src/classifier.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/naive_bayes.dir/src/classifier.cpp.o -c /home/lucho/Projects/SelfDrivingCar/naive_bayes/src/classifier.cpp

CMakeFiles/naive_bayes.dir/src/classifier.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/naive_bayes.dir/src/classifier.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lucho/Projects/SelfDrivingCar/naive_bayes/src/classifier.cpp > CMakeFiles/naive_bayes.dir/src/classifier.cpp.i

CMakeFiles/naive_bayes.dir/src/classifier.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/naive_bayes.dir/src/classifier.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lucho/Projects/SelfDrivingCar/naive_bayes/src/classifier.cpp -o CMakeFiles/naive_bayes.dir/src/classifier.cpp.s

CMakeFiles/naive_bayes.dir/src/classifier.cpp.o.requires:

.PHONY : CMakeFiles/naive_bayes.dir/src/classifier.cpp.o.requires

CMakeFiles/naive_bayes.dir/src/classifier.cpp.o.provides: CMakeFiles/naive_bayes.dir/src/classifier.cpp.o.requires
	$(MAKE) -f CMakeFiles/naive_bayes.dir/build.make CMakeFiles/naive_bayes.dir/src/classifier.cpp.o.provides.build
.PHONY : CMakeFiles/naive_bayes.dir/src/classifier.cpp.o.provides

CMakeFiles/naive_bayes.dir/src/classifier.cpp.o.provides.build: CMakeFiles/naive_bayes.dir/src/classifier.cpp.o


# Object files for target naive_bayes
naive_bayes_OBJECTS = \
"CMakeFiles/naive_bayes.dir/src/main.cpp.o" \
"CMakeFiles/naive_bayes.dir/src/classifier.cpp.o"

# External object files for target naive_bayes
naive_bayes_EXTERNAL_OBJECTS =

naive_bayes: CMakeFiles/naive_bayes.dir/src/main.cpp.o
naive_bayes: CMakeFiles/naive_bayes.dir/src/classifier.cpp.o
naive_bayes: CMakeFiles/naive_bayes.dir/build.make
naive_bayes: CMakeFiles/naive_bayes.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lucho/Projects/SelfDrivingCar/naive_bayes/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable naive_bayes"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/naive_bayes.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/naive_bayes.dir/build: naive_bayes

.PHONY : CMakeFiles/naive_bayes.dir/build

CMakeFiles/naive_bayes.dir/requires: CMakeFiles/naive_bayes.dir/src/main.cpp.o.requires
CMakeFiles/naive_bayes.dir/requires: CMakeFiles/naive_bayes.dir/src/classifier.cpp.o.requires

.PHONY : CMakeFiles/naive_bayes.dir/requires

CMakeFiles/naive_bayes.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/naive_bayes.dir/cmake_clean.cmake
.PHONY : CMakeFiles/naive_bayes.dir/clean

CMakeFiles/naive_bayes.dir/depend:
	cd /home/lucho/Projects/SelfDrivingCar/naive_bayes/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lucho/Projects/SelfDrivingCar/naive_bayes /home/lucho/Projects/SelfDrivingCar/naive_bayes /home/lucho/Projects/SelfDrivingCar/naive_bayes/build /home/lucho/Projects/SelfDrivingCar/naive_bayes/build /home/lucho/Projects/SelfDrivingCar/naive_bayes/build/CMakeFiles/naive_bayes.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/naive_bayes.dir/depend

