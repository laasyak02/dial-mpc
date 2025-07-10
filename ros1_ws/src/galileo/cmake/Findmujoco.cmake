# Check if the mujoco library is available
set(MUJOCO_ROOT_PATHS
  "/root/.mujoco/mujoco-3.3.1"
  "/usr/local"
  "/usr"
)

# Find include directory
find_path(mujoco_INCLUDE_DIRS
  NAMES mujoco/mujoco.h
  PATHS ${MUJOCO_ROOT_PATHS}
  PATH_SUFFIXES include
  DOC "MuJoCo include directory"
)

# Find library
find_library(mujoco_LIBRARIES
  NAMES mujoco libmujoco.so
  PATHS ${MUJOCO_ROOT_PATHS}
  PATH_SUFFIXES lib
  DOC "MuJoCo library"
)
# Set mujoco_FOUND to TRUE if both the include directory and library are found
if (mujoco_INCLUDE_DIRS AND mujoco_LIBRARIES)
    set(mujoco_FOUND TRUE)
else ()
    set(mujoco_FOUND FALSE)
endif ()

# Provide the version of mujoco if it's available
find_file(mujoco_VERSION_FILE NAMES version.txt PATHS ${MUJOCO_ROOT_PATHS} PATH_SUFFIXES lib/cmake/mujoco)
if (mujoco_VERSION_FILE)
    file(STRINGS ${mujoco_VERSION_FILE} mujoco_VERSION)
endif ()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(mujoco DEFAULT_MSG mujoco_LIBRARIES mujoco_INCLUDE_DIRS)

if (mujoco_FOUND)
    # Include directories for the mujoco library
    set(mujoco_INCLUDE_DIRS ${mujoco_INCLUDE_DIRS})

    # Library file for mujoco
    set(mujoco_LIBRARIES ${mujoco_LIBRARIES})
endif ()