QT       += core gui
QT += charts

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17
INCLUDEPATH += onnxruntime-win-x64-gpu-1.11.1/include/
LIBS += -lonnxruntime-win-x64-gpu-1.11.1/lib/*.lib
#LIBS += -lonnxruntime -onnxruntime_providers_shared

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    load_onnx.cpp \
    main.cpp \
    mainwindow.cpp

HEADERS += \
    load_onnx.h \
    mainwindow.h

FORMS += \
    mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
