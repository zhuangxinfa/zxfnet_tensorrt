#ifndef MY_MATMUL_PLUGIN_H
#define MY_MATMUL_PLUGIN_H
#include "NvInfer.h"
#include "NvUffParser.h"
#include <cassert>
#include <chrono>
#include <cudnn.h>
#include <iostream>
#include <map>
#include <string.h>
#include <unordered_map>
#include <vector>

#include "NvUtils.h"

using namespace nvuffparser;
using namespace nvinfer1;
#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

namespace common_function {


inline unsigned int elementSize(DataType t)
{
    switch (t)
    {
    case DataType::kINT32:
    case DataType::kFLOAT: return 4;
    case DataType::kHALF: return 2;
    case DataType::kINT8: return 1;
    }
    return 0;
}

// Return m rounded up to nearest multiple of n
inline int roundUp(int m, int n)
{
    return ((m + n - 1) / n) * n;
}
inline int getC(const Dims& d)
{
    return d.nbDims >= 3 ? d.d[d.nbDims - 3] : 1;
}

inline int getH(const Dims& d)
{
    return d.nbDims >= 2 ? d.d[d.nbDims - 2] : 1;
}

inline int getW(const Dims& d)
{
    return d.nbDims >= 1 ? d.d[d.nbDims - 1] : 1;
}
struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};
}
struct PoolParameters
{
    // Input dimensions
    int mC, mH, mW;
    // Output dimensions
    int mP, mQ;
    // Kernel size
    int mR, mS;
    // Stride
    int mU, mV;
    // Padding
    int pH, pW;
    // Pooling Function
    PoolingType pType;
};
/**
class SampleUffPluginV2Ext
{
public:
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, InferDeleter>;

    explicit SampleUffPluginV2Ext(const UffSampleParams& params)
        : mParams(params)
    {
    }

    //!
    //! \brief Creates the network, configures the builder and creates the network engine
    //!
    bool build();
    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

    //!
    //! \brief Used to clean up any state created in the sample class
    //!
    bool teardown();
private:
    SampleUniquePtr<nvinfer1::ICudaEngine> mEngine;
    samplesCommon::UffSampleParams mParams;
};
**/
class MyMatMulPlugin : public IPluginV2
{
public:
    MyMatMulPlugin(const PluginFieldCollection& fc);

    MyMatMulPlugin(const void* data, size_t length);

    // It makes no sense to construct UffPoolPluginV2 without arguments.
    MyMatMulPlugin()=delete;

    virtual ~MyMatMulPlugin() {}

public:
    //Get the number of outputs from the layer.
    int getNbOutputs() const override;
    //Get the dimension of an output tensor.
    /**
        index	The index of the output tensor.
        inputs	The input tensors.
        nbInputDims	The number of input tensors.
    **/
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;
    bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const override;
    void configureWithFormat(const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims, int nbOutputs, nvinfer1::DataType type, nvinfer1::PluginFormat format, int maxBatchSize) override;

    //Initialize the layer for execution. This is called when the engine is created.
    int initialize() override;
    //Release resources acquired during plugin layer initialization. This is called when the engine is destroyed.
    void terminate() override;

    //Find the workspace size required by the layer.
    //This function is called during engine startup, after initialize(). The workspace size returned should be sufficient for any batch size up to the maximum.
    size_t getWorkspaceSize(int maxBatchSize) const override;
    //Execute the layer.
    int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    //Find the size of the serialization buffer required.
    size_t getSerializationSize() const override;

    //Serialize the layer.
    //buffer: A pointer to a buffer to serialize data. Size of buffer must be equal to value returned by getSerializationSize.
    void serialize(void* buffer) const override;

    //Configure the layer.
    //This function is called by the builder prior to initialize(). It provides an opportunity for the layer to make algorithm choices on the basis of I/O PluginTensorDesc and the maximum batch size.
    // void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) override;

    //! The combination of kLINEAR + kINT8/kHALF/kFLOAT is supported.
    // bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const override;

    // DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const override;
    //	Return the plugin type. Should match the plugin name returned by the corresponding plugin creator.
    const char* getPluginType() const override;
    //Return the plugin version. Should match the plugin version returned by the corresponding plugin creator.
    const char* getPluginVersion() const override;
    void destroy() override;
    
    IPluginV2 * clone() const override;

    //Set the namespace that this plugin object belongs to. Ideally, all plugin objects from the same plugin library should have the same namespace.
    void setPluginNamespace(const char* libNamespace) override;
    const char* getPluginNamespace() const override;
    // bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

    // bool canBroadcastInputAcrossBatch(int inputIndex) const override;
private:
    template <typename T>
    void write(char*& buffer, const T& val) const
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template <typename T>
    T read(const char*& buffer) const
    {
        T val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
        return val;
    }

    // void copyDeviceInputToFP32(const void* src, void*& dst);

    // void copyDeviceToInt8Output(const void* src, void* dst);
private:
    cudnnHandle_t mCudnn;
    cudnnTensorDescriptor_t mSrcDescriptor, mDstDescriptor;//cudnnTensorDescriptor_t is a pointer to an opaque structure holding the description of a generic n-D dataset.
    //cudnnPoolingDescriptor_t mPoolingDesc;//是指向包含池操作描述的不透明结构的指针。
    //PoolParameters mPoolingParams;
   // cudnnPoolingMode_t mMode;
    DataType mDataType;

    Dims mInputDims;
    Dims mOutputDims;
    float mInHostScale{-1.0f};
    float mOutHostScale{-1.0f};
    std::string mNamespace;
};

class UffPoolPluginV2Creator : public IPluginCreator
{
public:
    const char* getPluginName() const override;
    const char* getPluginVersion() const override;

    //Return a list of fields that needs to be passed to createPlugin.
    const PluginFieldCollection* getFieldNames() override;

    //Return a plugin object. Return nullptr in case of error.
    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    //Called during deserialization of plugin layer. Return a plugin object.
    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

    //Set the namespace of the plugin creator based on the plugin library it belongs to. This can be set while registering the plugin creator.
    void setPluginNamespace(const char* libNamespace) override;
    const char* getPluginNamespace() const override;
private:
    std::string mNamespace;
    std::string mPluginName;
    PluginFieldCollection mFieldCollection{0, nullptr};
};

#endif // MY_MATMUL_PLUGIN_H
