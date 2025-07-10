# # Check if the mujoco library is available
# find_path(mujoco_INCLUDE_DIRS NAMES mujoco/mujoco.h PATHS /root/.mujoco/mujoco-3.3.1/include)

# # Check if the mujoco library is available
# find_library(mujoco_LIBRARIES NAMES libmujoco.so PATHS /root/.mujoco/mujoco-3.3.1/lib)

# # Set mujoco_FOUND to TRUE if both the include directory and library are found
# if (mujoco_INCLUDE_DIRS AND mujoco_LIBRARIES)
#     set(mujoco_FOUND TRUE)
# else ()
#     set(mujoco_FOUND FALSE)
# endif ()

# # Provide the version of mujoco if it's available
# find_file(mujoco_VERSION_FILE NAMES version.txt PATHS /root/.mujoco/mujoco210)
# if (mujoco_VERSION_FILE)
#     file(STRINGS ${mujoco_VERSION_FILE} mujoco_VERSION)
# endif ()

# include(FindPackageHandleStandardArgs)
# find_package_handle_standard_args(mujoco DEFAULT_MSG mujoco_LIBRARIES mujoco_INCLUDE_DIRS)

# if (mujoco_FOUND)
#     # Include directories for the mujoco library
#     set(mujoco_INCLUDE_DIRS ${mujoco_INCLUDE_DIRS})

#     # Library file for mujoco
#     set(mujoco_LIBRARIES ${mujoco_LIBRARIES})
# endif ()

# FindMujoco.cmake
# Locate the MuJoCo library and headers

# Define search paths
set(MUJOCO_ROOT_PATHS
  "/root/.mujoco/mujoco-3.3.1"
  "/usr/local"
  "/usr"
)

# Find include directory
find_path(MUJOCO_INCLUDE_DIR
  NAMES mujoco/mujoco.h
  PATHS ${MUJOCO_ROOT_PATHS}
  PATH_SUFFIXES include
  DOC "MuJoCo include directory"
)

# Find library
find_library(MUJOCO_LIBRARY
  NAMES mujoco libmujoco.so
  PATHS ${MUJOCO_ROOT_PATHS}
  PATH_SUFFIXES lib
  DOC "MuJoCo library"
)

# Set variables for standard find_package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(mujoco DEFAULT_MSG
  MUJOCO_INCLUDE_DIR MUJOCO_LIBRARY)

if(MUJOCO_FOUND)
  set(mujoco_INCLUDE_DIRS ${MUJOCO_INCLUDE_DIR})
  set(mujoco_LIBRARIES ${MUJOCO_LIBRARY})
  
  message(STATUS "Found MuJoCo:")
  message(STATUS "  Include: ${mujoco_INCLUDE_DIRS}")
  message(STATUS "  Library: ${mujoco_LIBRARIES}")
endif()

mark_as_advanced(MUJOCO_INCLUDE_DIR MUJOCO_LIBRARY)