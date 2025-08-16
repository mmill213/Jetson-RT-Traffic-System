# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_detections_img_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED detections_img_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(detections_img_FOUND FALSE)
  elseif(NOT detections_img_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(detections_img_FOUND FALSE)
  endif()
  return()
endif()
set(_detections_img_CONFIG_INCLUDED TRUE)

# output package information
if(NOT detections_img_FIND_QUIETLY)
  message(STATUS "Found detections_img: 0.0.0 (${detections_img_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'detections_img' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${detections_img_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(detections_img_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${detections_img_DIR}/${_extra}")
endforeach()
