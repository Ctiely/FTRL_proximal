# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.9

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
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/clytie/Documents/FTRL

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/clytie/Documents/FTRL/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/FTRL.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/FTRL.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/FTRL.dir/flags.make

CMakeFiles/FTRL.dir/src/dataset.cpp.o: CMakeFiles/FTRL.dir/flags.make
CMakeFiles/FTRL.dir/src/dataset.cpp.o: ../src/dataset.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/clytie/Documents/FTRL/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/FTRL.dir/src/dataset.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/FTRL.dir/src/dataset.cpp.o -c /Users/clytie/Documents/FTRL/src/dataset.cpp

CMakeFiles/FTRL.dir/src/dataset.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FTRL.dir/src/dataset.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/clytie/Documents/FTRL/src/dataset.cpp > CMakeFiles/FTRL.dir/src/dataset.cpp.i

CMakeFiles/FTRL.dir/src/dataset.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FTRL.dir/src/dataset.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/clytie/Documents/FTRL/src/dataset.cpp -o CMakeFiles/FTRL.dir/src/dataset.cpp.s

CMakeFiles/FTRL.dir/src/dataset.cpp.o.requires:

.PHONY : CMakeFiles/FTRL.dir/src/dataset.cpp.o.requires

CMakeFiles/FTRL.dir/src/dataset.cpp.o.provides: CMakeFiles/FTRL.dir/src/dataset.cpp.o.requires
	$(MAKE) -f CMakeFiles/FTRL.dir/build.make CMakeFiles/FTRL.dir/src/dataset.cpp.o.provides.build
.PHONY : CMakeFiles/FTRL.dir/src/dataset.cpp.o.provides

CMakeFiles/FTRL.dir/src/dataset.cpp.o.provides.build: CMakeFiles/FTRL.dir/src/dataset.cpp.o


CMakeFiles/FTRL.dir/src/model.cpp.o: CMakeFiles/FTRL.dir/flags.make
CMakeFiles/FTRL.dir/src/model.cpp.o: ../src/model.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/clytie/Documents/FTRL/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/FTRL.dir/src/model.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/FTRL.dir/src/model.cpp.o -c /Users/clytie/Documents/FTRL/src/model.cpp

CMakeFiles/FTRL.dir/src/model.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FTRL.dir/src/model.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/clytie/Documents/FTRL/src/model.cpp > CMakeFiles/FTRL.dir/src/model.cpp.i

CMakeFiles/FTRL.dir/src/model.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FTRL.dir/src/model.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/clytie/Documents/FTRL/src/model.cpp -o CMakeFiles/FTRL.dir/src/model.cpp.s

CMakeFiles/FTRL.dir/src/model.cpp.o.requires:

.PHONY : CMakeFiles/FTRL.dir/src/model.cpp.o.requires

CMakeFiles/FTRL.dir/src/model.cpp.o.provides: CMakeFiles/FTRL.dir/src/model.cpp.o.requires
	$(MAKE) -f CMakeFiles/FTRL.dir/build.make CMakeFiles/FTRL.dir/src/model.cpp.o.provides.build
.PHONY : CMakeFiles/FTRL.dir/src/model.cpp.o.provides

CMakeFiles/FTRL.dir/src/model.cpp.o.provides.build: CMakeFiles/FTRL.dir/src/model.cpp.o


CMakeFiles/FTRL.dir/main.cpp.o: CMakeFiles/FTRL.dir/flags.make
CMakeFiles/FTRL.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/clytie/Documents/FTRL/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/FTRL.dir/main.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/FTRL.dir/main.cpp.o -c /Users/clytie/Documents/FTRL/main.cpp

CMakeFiles/FTRL.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FTRL.dir/main.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/clytie/Documents/FTRL/main.cpp > CMakeFiles/FTRL.dir/main.cpp.i

CMakeFiles/FTRL.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FTRL.dir/main.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/clytie/Documents/FTRL/main.cpp -o CMakeFiles/FTRL.dir/main.cpp.s

CMakeFiles/FTRL.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/FTRL.dir/main.cpp.o.requires

CMakeFiles/FTRL.dir/main.cpp.o.provides: CMakeFiles/FTRL.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/FTRL.dir/build.make CMakeFiles/FTRL.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/FTRL.dir/main.cpp.o.provides

CMakeFiles/FTRL.dir/main.cpp.o.provides.build: CMakeFiles/FTRL.dir/main.cpp.o


# Object files for target FTRL
FTRL_OBJECTS = \
"CMakeFiles/FTRL.dir/src/dataset.cpp.o" \
"CMakeFiles/FTRL.dir/src/model.cpp.o" \
"CMakeFiles/FTRL.dir/main.cpp.o"

# External object files for target FTRL
FTRL_EXTERNAL_OBJECTS =

FTRL: CMakeFiles/FTRL.dir/src/dataset.cpp.o
FTRL: CMakeFiles/FTRL.dir/src/model.cpp.o
FTRL: CMakeFiles/FTRL.dir/main.cpp.o
FTRL: CMakeFiles/FTRL.dir/build.make
FTRL: CMakeFiles/FTRL.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/clytie/Documents/FTRL/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable FTRL"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/FTRL.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/FTRL.dir/build: FTRL

.PHONY : CMakeFiles/FTRL.dir/build

CMakeFiles/FTRL.dir/requires: CMakeFiles/FTRL.dir/src/dataset.cpp.o.requires
CMakeFiles/FTRL.dir/requires: CMakeFiles/FTRL.dir/src/model.cpp.o.requires
CMakeFiles/FTRL.dir/requires: CMakeFiles/FTRL.dir/main.cpp.o.requires

.PHONY : CMakeFiles/FTRL.dir/requires

CMakeFiles/FTRL.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/FTRL.dir/cmake_clean.cmake
.PHONY : CMakeFiles/FTRL.dir/clean

CMakeFiles/FTRL.dir/depend:
	cd /Users/clytie/Documents/FTRL/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/clytie/Documents/FTRL /Users/clytie/Documents/FTRL /Users/clytie/Documents/FTRL/cmake-build-debug /Users/clytie/Documents/FTRL/cmake-build-debug /Users/clytie/Documents/FTRL/cmake-build-debug/CMakeFiles/FTRL.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/FTRL.dir/depend

