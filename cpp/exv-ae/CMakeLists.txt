cmake_minimum_required(VERSION 3.15)
project(ExvAE CXX)

# After Effects AEIO plugin (.aex) for EXV — reads/writes .exv (a still is a 1-frame .exv)
# via the pure-Rust codec through exv_capi.dll (crates/exv-capi).
# Windows-focused (PiPL pipeline + MediaCore deploy); a macOS path can be added later.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# AE SDK (override with -DAE_SDK_PATH=...). Defaults to the sibling repo's SDK.
if(NOT DEFINED AE_SDK_PATH)
	set(AE_SDK_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../vfx-aeio/vfx-codec/cpp/sdk/ae25.6_61.64bit.AfterEffectsSDK/Examples"
	    CACHE PATH "Path to the After Effects SDK Examples folder")
endif()

# exv-capi: C ABI header (the DLL is LoadLibrary'd at run time; bootstrap.py deploys it next to the .aex).
set(EXV_CAPI_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../../docs" CACHE PATH "dir with exv-capi.h")
# Runtime model: exv_capi.dll is LoadLibrary'd at run time (no import lib).

set(PLUGIN_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/ExvAE.cpp)
set(UTIL_SOURCES
	${AE_SDK_PATH}/Util/AEGP_SuiteHandler.cpp
	${AE_SDK_PATH}/Util/MissingSuiteError.cpp
)
set(AESDK_INCLUDE
	${AE_SDK_PATH}/Headers ${AE_SDK_PATH}/Headers/SP
	${AE_SDK_PATH}/Util ${AE_SDK_PATH}/Resources
)
include_directories(${AESDK_INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR}/src
                    ${CMAKE_CURRENT_SOURCE_DIR}/include ${EXV_CAPI_INCLUDE})

add_library(${PROJECT_NAME} MODULE ${PLUGIN_SOURCES} ${UTIL_SOURCES})

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
	add_compile_definitions(_CRT_SECURE_NO_WARNINGS MSWindows)
	set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".aex")

	# PiPL resource pipeline: cl /EP -> PiPLtool -> .rc.
	get_filename_component(AFX_PIPLTOOL ${AE_SDK_PATH}/Resources/PiPLtool.exe ABSOLUTE CACHE)
	add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.rr
		COMMAND cl /I "${AE_SDK_PATH}/Headers" /I "${AE_SDK_PATH}/Headers/SP" /I "${AE_SDK_PATH}/Util" /I "${AE_SDK_PATH}/Resources"
			/EP "${CMAKE_CURRENT_SOURCE_DIR}/src/ExvAE_PiPL.r" > "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.rr"
		COMMENT "Preprocessing PiPL")
	add_custom_command(DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.rr
		OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.rrc
		COMMAND ${AFX_PIPLTOOL} ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.rr ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.rrc
		COMMENT "Running PiPLtool")
	add_custom_command(DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.rrc
		OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.rc
		COMMAND cl /D "MSWindows" /EP ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.rrc > ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.rc
		COMMENT "Generating rc")
	target_sources(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.rc)

	# NO deploy step here: `python bootstrap.py ae|deploy` is the ONE place that copies ExvAE.aex +
	# exv_capi.dll into AE MediaCore / Support Files (it refuses while a MediaCore host is running).
elseif(APPLE)
	add_compile_definitions(MacOS)
	set_target_properties(${PROJECT_NAME} PROPERTIES
		BUNDLE TRUE
		BUNDLE_EXTENSION "plugin"
		MACOSX_BUNDLE_BUNDLE_NAME "${PROJECT_NAME}"
		MACOSX_BUNDLE_INFO_STRING "EXV After Effects AEIO Plugin"
		MACOSX_BUNDLE_GUI_IDENTIFIER "com.exr-rs.ExvAE"
	)
	find_library(COCOA_FRAMEWORK Cocoa REQUIRED)
	find_library(COREFOUNDATION_FRAMEWORK CoreFoundation REQUIRED)
	target_link_libraries(${PROJECT_NAME} PRIVATE ${COCOA_FRAMEWORK} ${COREFOUNDATION_FRAMEWORK})

	find_program(REZ_TOOL Rez HINTS /usr/bin /Library/Developer/CommandLineTools/usr/bin)
	if(REZ_TOOL)
		set(RSRC_OUT "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.rsrc")
		add_custom_command(OUTPUT "${RSRC_OUT}"
			COMMAND "${REZ_TOOL}"
				-d SystemSevenOrLater=1
				-useDF
				-script Roman
				-i "${AE_SDK_PATH}/Headers"
				-i "${AE_SDK_PATH}/Headers/SP"
				-i "${AE_SDK_PATH}/Util"
				-i "${AE_SDK_PATH}/Resources"
				-o "${RSRC_OUT}"
				"${CMAKE_CURRENT_SOURCE_DIR}/src/ExvAE_PiPL.r"
			DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/ExvAE_PiPL.r"
			COMMENT "Compiling macOS PiPL resource via Rez")
		add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_BUNDLE_CONTENT_DIR:${PROJECT_NAME}>/Resources"
			COMMAND ${CMAKE_COMMAND} -E copy_if_different "${RSRC_OUT}" "$<TARGET_BUNDLE_CONTENT_DIR:${PROJECT_NAME}>/Resources/${PROJECT_NAME}.rsrc"
			DEPENDS "${RSRC_OUT}")
	endif()
endif()

# Pure-logic tests (no AE SDK): `bootstrap.py test` builds + runs them via ctest; by hand:
#   cmake --build . --target cache_key_test enc_opts_test && ctest -C Release
enable_testing()
function(exv_test name)
	add_executable(${name} EXCLUDE_FROM_ALL tests/${name}.cpp)
	target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${EXV_CAPI_INCLUDE})
	add_test(NAME ${name} COMMAND ${name})
endfunction()
exv_test(cache_key_test)
exv_test(enc_opts_test)
