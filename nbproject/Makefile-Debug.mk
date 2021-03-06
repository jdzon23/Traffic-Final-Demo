#
# Generated Makefile - do not edit!
#
# Edit the Makefile in the project folder instead (../Makefile). Each target
# has a -pre and a -post target defined where you can add customized code.
#
# This makefile implements configuration specific macros and targets.


# Environment
MKDIR=mkdir
CP=cp
GREP=grep
NM=nm
CCADMIN=CCadmin
RANLIB=ranlib
CC=gcc
CCC=g++
CXX=g++
FC=gfortran
AS=as

# Macros
CND_PLATFORM=MinGW_1-Windows
CND_DLIB_EXT=dll
CND_CONF=Debug
CND_DISTDIR=dist
CND_BUILDDIR=build

# Include project Makefile
include Makefile

# Object Directory
OBJECTDIR=${CND_BUILDDIR}/${CND_CONF}/${CND_PLATFORM}

# Object Files
OBJECTFILES= \
	${OBJECTDIR}/Blob.o \
	${OBJECTDIR}/Timer.o \
	${OBJECTDIR}/main.o


# C Compiler Flags
CFLAGS=

# CC Compiler Flags
CCFLAGS=
CXXFLAGS=

# Fortran Compiler Flags
FFLAGS=

# Assembler Flags
ASFLAGS=

# Link Libraries and Options
LDLIBSOPTIONS=-L/D/opencv/new_build/install/x86/mingw/lib /D/opencv/new_build/install/x86/mingw/lib/libopencv_calib3d320.dll.a /D/opencv/new_build/install/x86/mingw/lib/libopencv_core320.dll.a /D/opencv/new_build/install/x86/mingw/lib/libopencv_dnn320.dll.a /D/opencv/new_build/install/x86/mingw/lib/libopencv_features2d320.dll.a /D/opencv/new_build/install/x86/mingw/lib/libopencv_flann320.dll.a /D/opencv/new_build/install/x86/mingw/lib/libopencv_highgui320.dll.a /D/opencv/new_build/install/x86/mingw/lib/libopencv_imgcodecs320.dll.a /D/opencv/new_build/install/x86/mingw/lib/libopencv_imgproc320.dll.a /D/opencv/new_build/install/x86/mingw/lib/libopencv_ml320.dll.a /D/opencv/new_build/install/x86/mingw/lib/libopencv_objdetect320.dll.a /D/opencv/new_build/install/x86/mingw/lib/libopencv_photo320.dll.a /D/opencv/new_build/install/x86/mingw/lib/libopencv_shape320.dll.a /D/opencv/new_build/install/x86/mingw/lib/libopencv_stitching320.dll.a /D/opencv/new_build/install/x86/mingw/lib/libopencv_superres320.dll.a /D/opencv/new_build/install/x86/mingw/lib/libopencv_text320.dll.a /D/opencv/new_build/install/x86/mingw/lib/libopencv_video320.dll.a /D/opencv/new_build/install/x86/mingw/lib/libopencv_videoio320.dll.a /D/opencv/new_build/install/x86/mingw/lib/libopencv_videostab320.dll.a /D/opencv/new_build/install/x86/mingw/lib/libopencv_xfeatures2d320.dll.a /D/opencv/new_build/install/x86/mingw/lib/libopencv_ximgproc320.dll.a /D/opencv/new_build/install/x86/mingw/lib/libopencv_xobjdetect320.dll.a /D/opencv/new_build/install/x86/mingw/lib/libopencv_xphoto320.dll.a

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/trafficdemo.exe

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/trafficdemo.exe: /D/opencv/new_build/install/x86/mingw/lib/libopencv_calib3d320.dll.a

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/trafficdemo.exe: /D/opencv/new_build/install/x86/mingw/lib/libopencv_core320.dll.a

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/trafficdemo.exe: /D/opencv/new_build/install/x86/mingw/lib/libopencv_dnn320.dll.a

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/trafficdemo.exe: /D/opencv/new_build/install/x86/mingw/lib/libopencv_features2d320.dll.a

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/trafficdemo.exe: /D/opencv/new_build/install/x86/mingw/lib/libopencv_flann320.dll.a

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/trafficdemo.exe: /D/opencv/new_build/install/x86/mingw/lib/libopencv_highgui320.dll.a

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/trafficdemo.exe: /D/opencv/new_build/install/x86/mingw/lib/libopencv_imgcodecs320.dll.a

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/trafficdemo.exe: /D/opencv/new_build/install/x86/mingw/lib/libopencv_imgproc320.dll.a

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/trafficdemo.exe: /D/opencv/new_build/install/x86/mingw/lib/libopencv_ml320.dll.a

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/trafficdemo.exe: /D/opencv/new_build/install/x86/mingw/lib/libopencv_objdetect320.dll.a

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/trafficdemo.exe: /D/opencv/new_build/install/x86/mingw/lib/libopencv_photo320.dll.a

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/trafficdemo.exe: /D/opencv/new_build/install/x86/mingw/lib/libopencv_shape320.dll.a

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/trafficdemo.exe: /D/opencv/new_build/install/x86/mingw/lib/libopencv_stitching320.dll.a

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/trafficdemo.exe: /D/opencv/new_build/install/x86/mingw/lib/libopencv_superres320.dll.a

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/trafficdemo.exe: /D/opencv/new_build/install/x86/mingw/lib/libopencv_text320.dll.a

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/trafficdemo.exe: /D/opencv/new_build/install/x86/mingw/lib/libopencv_video320.dll.a

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/trafficdemo.exe: /D/opencv/new_build/install/x86/mingw/lib/libopencv_videoio320.dll.a

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/trafficdemo.exe: /D/opencv/new_build/install/x86/mingw/lib/libopencv_videostab320.dll.a

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/trafficdemo.exe: /D/opencv/new_build/install/x86/mingw/lib/libopencv_xfeatures2d320.dll.a

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/trafficdemo.exe: /D/opencv/new_build/install/x86/mingw/lib/libopencv_ximgproc320.dll.a

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/trafficdemo.exe: /D/opencv/new_build/install/x86/mingw/lib/libopencv_xobjdetect320.dll.a

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/trafficdemo.exe: /D/opencv/new_build/install/x86/mingw/lib/libopencv_xphoto320.dll.a

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/trafficdemo.exe: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/trafficdemo ${OBJECTFILES} ${LDLIBSOPTIONS}

${OBJECTDIR}/Blob.o: Blob.cpp
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -g -I/D/opencv/new_build/install/include -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/Blob.o Blob.cpp

${OBJECTDIR}/Timer.o: Timer.cpp
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -g -I/D/opencv/new_build/install/include -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/Timer.o Timer.cpp

${OBJECTDIR}/main.o: main.cpp
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -g -I/D/opencv/new_build/install/include -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/main.o main.cpp

# Subprojects
.build-subprojects:

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}

# Subprojects
.clean-subprojects:

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
