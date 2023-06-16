#include"load_onnx.h"
#include<assert.h>
#include<cmath>
#include<stdlib.h>
#include<stdio.h>
#include<QDebug>
#include<vector>
#include<iostream>
#include<math.h>
#include"onnxruntime_c_api.h"
#include"cpu_provider_factory.h"
#pragma comment(lib,".\\onnxruntime-win-x64-gpu-1.11.1\\lib\\onnxruntime.lib")
#pragma comment(lib,".\\onnxruntime-win-x64-gpu-1.11.1\\lib\\onnxruntime_providers_shared.lib")
const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
//Below is the prepare operation

//*****************************************************************************
// helper function to check for status
void CheckStatus(OrtStatus* status)
{
    if (status != NULL) {
        qDebug()<<"error occured!";
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "%s\n", msg);
        g_ort->ReleaseStatus(status);
        exit(1);
    }
}

// softmax method
float* softmax(float* modeloutput)
{
    int class_number;
    float out[3];
    class_number = 3;
    int i = 0;
    float sum;
    sum = 0;
    for (i = 0; i < class_number; i++)
    {
        sum = sum + exp(modeloutput[i]);
    }
    for (i = 0; i < class_number; i++)
    {
        out[i] = exp(modeloutput[i]) / sum;
    }
    return out;
}

load_onnx::load_onnx(QObject *parent)
{

}

QList<float> load_onnx::ResNet(QStringList input)
{
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtEnv* env;
    CheckStatus(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));
    // set up parameters
    OrtSessionOptions* session_options;
    CheckStatus(g_ort->CreateSessionOptions(&session_options));
    // Setting the number of threads
    g_ort->SetIntraOpNumThreads(session_options, 1);
    // Setting the optimization level
    g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_BASIC);
    CheckStatus(OrtSessionOptionsAppendExecutionProvider_CPU(session_options, 0));
    OrtSession* session;
    const wchar_t* model_path = L"covid_resnet50.onnx";
    qDebug()<<"Using Onnxruntime C API\n";
    // create a session
    CheckStatus(g_ort->CreateSession(env, model_path, session_options, &session));

    //*************************************************************************
    // print model input layer (node names, types, shape etc.)
    size_t num_input_nodes;
    OrtStatus* status;
    OrtAllocator* allocator;
    CheckStatus(g_ort->GetAllocatorWithDefaultOptions(&allocator));

    // print number of model input nodes
    status = g_ort->SessionGetInputCount(session, &num_input_nodes);
    std::vector<const char*> input_node_names(num_input_nodes);
    std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
    // Otherwise need vector<vector<>>
    qDebug("Number of inputs = %zu\n", num_input_nodes);

    // iterate over all input nodes
    for (size_t i = 0; i < num_input_nodes; i++) {
        // print input node names
        char* input_name;
        status = g_ort->SessionGetInputName(session, i, allocator, &input_name);
        qDebug("Input %zu : name=%s\n", i, input_name);
        input_node_names[i] = input_name;

        // print input node types
        OrtTypeInfo* typeinfo;
        status = g_ort->SessionGetInputTypeInfo(session, i, &typeinfo);
        const OrtTensorTypeAndShapeInfo* tensor_info;
        CheckStatus(g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));
        ONNXTensorElementDataType type;
        CheckStatus(g_ort->GetTensorElementType(tensor_info, &type));
        qDebug("Input %zu : type=%d\n", i, type);

        // print input shapes/dims
        size_t num_dims;
        CheckStatus(g_ort->GetDimensionsCount(tensor_info, &num_dims));
        qDebug("Input %zu : num_dims=%zu\n", i, num_dims);
        input_node_dims.resize(num_dims);
        g_ort->GetDimensions(tensor_info, (int64_t*)input_node_dims.data(), num_dims);
        input_node_dims[0] = 1;
        input_node_dims[1] = 1;
        input_node_dims[2] = 900;
        for (size_t j = 0; j < num_dims; j++)
        {
            qDebug("Input %zu : dim %zu=%jd\n", i, j, input_node_dims[j]);

        }

        g_ort->ReleaseTypeInfo(typeinfo);
    }


    // use OrtGetTensorShapeElementCount() to get official size!
    size_t input_tensor_size = 900;
    std::vector<float> input_tensor_values(input_tensor_size);
    std::vector<const char*> output_node_names = { "output" }; // 输出节点

    // initialize input data with values in [0.0, 1.0]
    // Here, direct assignment is used
    for (size_t i = 0; i < input_tensor_size; i++)
        input_tensor_values[i] = input[i].toFloat();

    // create input tensor object from data values
    OrtMemoryInfo* memory_info;
    CheckStatus(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
    OrtValue* input_tensor = NULL;
    g_ort->CreateTensorAsOrtValue(allocator, input_node_dims.data(), 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);
    CheckStatus(g_ort->CreateTensorWithDataAsOrtValue(memory_info, input_tensor_values.data(), input_tensor_size * sizeof(float), input_node_dims.data(), 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));
    int is_tensor;
    CheckStatus(g_ort->IsTensor(input_tensor, &is_tensor));
    assert(is_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);

    // score model & input tensor, get back output tensor
    OrtValue* output_tensor = NULL;
    //while (true)
    //{
    CheckStatus(g_ort->Run(session, NULL, input_node_names.data(), (const OrtValue* const*)&input_tensor, 1, output_node_names.data(), 1, &output_tensor));
    CheckStatus(g_ort->IsTensor(output_tensor, &is_tensor));
    assert(is_tensor);

    // Get pointer to output tensor float values
    float* floatarr;
    CheckStatus(g_ort->GetTensorMutableData(output_tensor, (void**)&floatarr));
    assert(std::abs(floatarr[0] - 0.000045) < 1e-6);

    // score the model, and print scores for first 3 classes
    QList<float> output_result;
    for (int i = 0; i < 5; i++) {
        qDebug("Score for class [%d] =  %f\n", i, floatarr[i]);
        output_result.append(floatarr[i]);
    }
    //}
    qDebug()<<"model=" << model_path;
//    auto out = softmax(floatarr);
//    for (int i = 0; i < 3; i++)
//    {
//        qDebug("accuracy for class [%d] = %lf\n", i, out[i]);
//        output[i] = out[i];
//    }
    g_ort->ReleaseValue(output_tensor);
    g_ort->ReleaseValue(input_tensor);
    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseEnv(env);
    qDebug("Done!\n");
    return output_result;
}

QList<float> load_onnx::inferenceModel(QList<float> spectrum, QString fileName, int classes)
{
    int spectrum_length = spectrum.length();
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtEnv* env;
    CheckStatus(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));
    // set up parameters
    OrtSessionOptions* session_options;
    CheckStatus(g_ort->CreateSessionOptions(&session_options));
    // Setting the number of threads
    g_ort->SetIntraOpNumThreads(session_options, 1);
    // Setting the optimization level
    g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_BASIC);
    CheckStatus(OrtSessionOptionsAppendExecutionProvider_CPU(session_options, 0));
    OrtSession* session;
    const wchar_t* model_path = reinterpret_cast<const wchar_t *>(fileName.utf16());
    qDebug()<<"Using Onnxruntime C API\n";
    // create a session
    CheckStatus(g_ort->CreateSession(env, model_path, session_options, &session));

    //*************************************************************************
    // print model input layer (node names, types, shape etc.)
    size_t num_input_nodes;
    OrtStatus* status;
    OrtAllocator* allocator;
    CheckStatus(g_ort->GetAllocatorWithDefaultOptions(&allocator));

    // print number of model input nodes
    status = g_ort->SessionGetInputCount(session, &num_input_nodes);
    std::vector<const char*> input_node_names(num_input_nodes);
    std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
    // Otherwise need vector<vector<>>
    qDebug("Number of inputs = %zu\n", num_input_nodes);

    // iterate over all input nodes
    for (size_t i = 0; i < num_input_nodes; i++) {
        // print input node names
        char* input_name;
        status = g_ort->SessionGetInputName(session, i, allocator, &input_name);
        qDebug("Input %zu : name=%s\n", i, input_name);
        input_node_names[i] = input_name;

        // print input node types
        OrtTypeInfo* typeinfo;
        status = g_ort->SessionGetInputTypeInfo(session, i, &typeinfo);
        const OrtTensorTypeAndShapeInfo* tensor_info;
        CheckStatus(g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));
        ONNXTensorElementDataType type;
        CheckStatus(g_ort->GetTensorElementType(tensor_info, &type));
        qDebug("Input %zu : type=%d\n", i, type);

        // print input shapes/dims
        size_t num_dims;
        CheckStatus(g_ort->GetDimensionsCount(tensor_info, &num_dims));
        qDebug("Input %zu : num_dims=%zu\n", i, num_dims);
        input_node_dims.resize(num_dims);
        g_ort->GetDimensions(tensor_info, (int64_t*)input_node_dims.data(), num_dims);
        input_node_dims[0] = 1;
        input_node_dims[1] = 1;
        input_node_dims[2] = spectrum_length;
        for (size_t j = 0; j < num_dims; j++)
        {
            qDebug("Input %zu : dim %zu=%jd\n", i, j, input_node_dims[j]);
        }

        g_ort->ReleaseTypeInfo(typeinfo);
    }

    // use OrtGetTensorShapeElementCount() to get official size!
    size_t input_tensor_size = spectrum_length;
    std::vector<float> input_tensor_values(input_tensor_size);
    std::vector<const char*> output_node_names = { "output" }; // output node

    // Here, direct assignment is used
    for (size_t i = 0; i < input_tensor_size; i++)
        input_tensor_values[i] = spectrum[i];

    // create input tensor object from data values
    OrtMemoryInfo* memory_info;
    CheckStatus(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
    OrtValue* input_tensor = NULL;
    g_ort->CreateTensorAsOrtValue(allocator, input_node_dims.data(), 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);
    CheckStatus(g_ort->CreateTensorWithDataAsOrtValue(memory_info, input_tensor_values.data(), input_tensor_size * sizeof(float), input_node_dims.data(), 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));
    int is_tensor;
    CheckStatus(g_ort->IsTensor(input_tensor, &is_tensor));
    assert(is_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);

    // score model & input tensor, get back output tensor
    OrtValue* output_tensor = NULL;
    CheckStatus(g_ort->Run(session, NULL, input_node_names.data(), (const OrtValue* const*)&input_tensor, 1, output_node_names.data(), 1, &output_tensor));
    CheckStatus(g_ort->IsTensor(output_tensor, &is_tensor));
    assert(is_tensor);

    // Get pointer to output tensor float values
    float* floatarr;
    CheckStatus(g_ort->GetTensorMutableData(output_tensor, (void**)&floatarr));
    assert(std::abs(floatarr[0] - 0.000045) < 1e-6);

    // score the model, and print scores for first 3 classes
    QList<float> output_result;
    for (int i = 0; i < classes; i++) {
        qDebug("Score for class [%d] =  %f\n", i, floatarr[i]);
        output_result.append(floatarr[i]);
    }
    qDebug()<<"model=" << model_path;
    g_ort->ReleaseValue(output_tensor);
    g_ort->ReleaseValue(input_tensor);
    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseEnv(env);
    qDebug("Done!\n");
    return output_result;
}
