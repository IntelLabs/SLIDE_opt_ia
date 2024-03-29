CMAKE_MINIMUM_REQUIRED(VERSION 3.0 FATAL_ERROR)

PROJECT( HashDeepLearning )

SET( CMAKE_CXX_STANDARD 14 )

# cross-platform OpenMP find
IF(APPLE)
  # MESSAGE (STATUS "${OpenMP_LIBRARIES}")
  IF ("${OpenMP_LIBRARIES}" STREQUAL "")
    FIND_LIBRARY(OpenMP_LIBRARY
        NAMES omp
    )
    FIND_PATH(OpenMP_INCLUDE_DIR
        omp.h
    )
    MARK_AS_ADVANCED(OpenMP_LIBRARY OpenMP_INCLUDE_DIR)
    INCLUDE(FINDPACKAGEHANDLESTANDARDARGS)
    FIND_PACKAGE_HANDLE_STANDARD_ARGS(OpenMP DEFAULT_MSG 
        OpenMP_LIBRARY OpenMP_INCLUDE_DIR)
    IF (OpenMP_FOUND)
        # MESSAGE (STATUS "OpenMP FOUND")
        SET(OpenMP_LIBRARIES ${OpenMP_LIBRARY} CACHE STRING "Cache OpenMP Lib" FORCE)
        SET(OpenMP_INCLUDE_DIRS ${OpenMP_INCLUDE_DIR} CACHE STRING "Cache OpenMP Include" FORCE)
        SET(OpenMP_COMPILE_OPTIONS -XPREPROCESSOR -FOpenMP -LOMP)
        ADD_LIBRARY(OpenMP::OpenMP SHARED IMPORTED)
        SET_TARGET_PROPERTIES(OpenMP::OpenMP PROPERTIES
            IMPORTED_LOCATION ${OpenMP_LIBRARIES}
            INTERFACE_INCLUDE_DIRECTORIES "${OpenMP_INCLUDE_DIRS}"
            INTERFACE_COMPILE_OPTIONS "${OpenMP_COMPILE_OPTIONS}"
        )
        ADD_DEFINITIONS(-D_OPENMP="${OpenMP_CXX_VERSION}")
        SET( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    ENDIF()
  ENDIF()
ELSE()
  FIND_PACKAGE(OpenMP REQUIRED)
  SET( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
  SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
ENDIF()

# Enable ICC & AVX512 build
option(OPT_IA "With IA general optimization" ON)
option(OPT_AVX512 "With AVX512 optimization" OFF)
option(OPT_AVX512_BF16 "With AVX512 BF16 optimization" OFF)
# if (CMAKE_CXX_COMPILER MATCHES "icpc")
  set(__opt_flags "-std=c++14 -O2 -DNDEBUG")
  # list(APPEND __opt_flags "-xHost")
  # list(APPEND __opt_flags "-qopt-zmm-usage=high")
  # list(APPEND __opt_flags "-no-inline-max-size")
  # list(APPEND __opt_flags "-no-inline-max-total-size")
  if (OPT_IA)
    list(APPEND __opt_flags "-DOPT_IA=1")
  endif()
  if (OPT_AVX512)
    list(APPEND __opt_flags "-DOPT_AVX512=1")
  endif()
  if (OPT_AVX512_BF16)
    list(APPEND __opt_flags "-DOPT_AVX512_BF16=1")
  endif()
  add_definitions(${__opt_flags})
# endif()

# build dependencies
INCLUDE( ExternalProject )

FIND_PACKAGE( Git REQUIRED )

OPTION( USE_GIT_PROTOCOL "If behind a firewall turn this off to use https instead." OFF )

function(functionInstallExternalCMakeProject ep_name)
  ExternalProject_Get_Property(${ep_name} binary_dir)
  install(SCRIPT ${binary_dir}/cmake_install.cmake)
endfunction()

ExternalProject_Add( 
  ZLIB
  DEPENDS ""
  GIT_REPOSITORY https://github.com/madler/zlib.git
  GIT_TAG v1.2.11
  SOURCE_DIR ZLIB-source
  BINARY_DIR ZLIB-build
  UPDATE_COMMAND ""
  PATCH_COMMAND ""
  # INSTALL_COMMAND ""
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX:STRING=${PROJECT_BINARY_DIR}/ep
    -DINSTALL_BIN_DIR:STRING=${PROJECT_BINARY_DIR}/ep/bin
    -DINSTALL_INC_DIR:STRING=${PROJECT_BINARY_DIR}/ep/include
    -DINSTALL_LIB_DIR:STRING=${PROJECT_BINARY_DIR}/ep/lib
    -DINSTALL_MAN_DIR:STRING=${PROJECT_BINARY_DIR}/ep/share/man
    -DINSTALL_PKGCONFIG_DIR:STRING=${PROJECT_BINARY_DIR}/ep/share/pkgconfig
    -DCMAKE_BUILD_TYPE:STRING=Release
)
functionInstallExternalCMakeProject(ZLIB)

# set the expected zlib libraries
IF( WIN32 )
  SET( ZLIB_LIB_DEBUG ${PROJECT_BINARY_DIR}/ep/lib/zlibstaticd.lib )
  SET( ZLIB_LIB_RELEASE ${PROJECT_BINARY_DIR}/ep/lib/zlibstatic.lib )
ELSE()
  SET( ZLIB_LIB_DEBUG ${PROJECT_BINARY_DIR}/ep/lib/libz.a )
  SET( ZLIB_LIB_RELEASE ${PROJECT_BINARY_DIR}/ep/lib/libz.a )
ENDIF()

ExternalProject_Add( 
  CNPY
  DEPENDS ZLIB
  GIT_REPOSITORY https://github.com/sarthakpati/cnpy.git
  # GIT_TAG v1.2.11
  SOURCE_DIR CNPY-source
  BINARY_DIR CNPY-build
  UPDATE_COMMAND ""
  PATCH_COMMAND ""
  # INSTALL_COMMAND ""
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
    -DZLIB_INCLUDE_DIR:STRING=${PROJECT_BINARY_DIR}/ep/include
    -DZLIB_LIBRARY_DEBUG:STRING=${ZLIB_LIB_DEBUG}
    -DZLIB_LIBRARY_RELEASE:STRING=${ZLIB_LIB_RELEASE}
    -DCMAKE_INSTALL_PREFIX:STRING=${PROJECT_BINARY_DIR}/ep
    -DBUILD_SHARED_LIBS:BOOL=${BUILD_SHARED_LIBS} 
    -DCMAKE_BUILD_TYPE:STRING=Release
)
functionInstallExternalCMakeProject(CNPY)

INCLUDE_DIRECTORIES( ${PROJECT_BINARY_DIR}/ep/include )

# set CNPY lib path
IF( WIN32 )
  SET( CNPY_LIB ${PROJECT_BINARY_DIR}/ep/lib/cnpy.lib )
ELSE()
SET( CNPY_LIB ${PROJECT_BINARY_DIR}/ep/lib/libcnpy.a )
ENDIF()

# now build SLIDE
FILE( GLOB_RECURSE SLIDE_SOURCES "${PROJECT_SOURCE_DIR}/SLIDE/*.cpp" )
FILE( GLOB_RECURSE SLIDE_HEADERS "${PROJECT_SOURCE_DIR}/SLIDE/*.h" )

# add library to decouple compilation
ADD_LIBRARY( SLIDE_LIB ${SLIDE_HEADERS} ${SLIDE_SOURCES} )
ADD_DEPENDENCIES( SLIDE_LIB CNPY )
TARGET_LINK_LIBRARIES( SLIDE_LIB ${CNPY_LIB} )

# add executable
SET( SLIDE_EXE_NAME runme )
ADD_EXECUTABLE( ${SLIDE_EXE_NAME} ${PROJECT_SOURCE_DIR}/SLIDE/main.cpp )
ADD_DEPENDENCIES( ${SLIDE_EXE_NAME} SLIDE_LIB )
TARGET_LINK_LIBRARIES( 
  ${SLIDE_EXE_NAME} 
  SLIDE_LIB 
  ${CNPY_LIB} 
  ${ZLIB_LIB_RELEASE} ) # TBD: this should be changed to use ${ZLIB_LIBRARIES} for debug portability on Windows
INSTALL( TARGETS ${SLIDE_EXE_NAME} DESTINATION bin )
