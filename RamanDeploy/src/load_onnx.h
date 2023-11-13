#ifndef LOAD_ONNX_H
#define LOAD_ONNX_H

#include <QList>
#include <QObject>

class load_onnx
{
public:
    explicit load_onnx(QObject *parent = nullptr);

    QList<float> ResNet(QStringList input);

    QList<float> inferenceModel(QList<float> spectrum, QString fileName, int classes);

};

#endif // LOAD_ONNX_H
