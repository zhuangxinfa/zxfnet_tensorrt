#include"zxfnet.h"
#include<iostream>
#include "NvInfer.h"
#include "NvUffParser.h"
#include <cuda_runtime_api.h>
#include<string>
using namespace nvinfer1;
using namespace nvuffparser;
class Logger :public ILogger{
    void log(Severity severity, const char *msg)override{
        std::cout<<msg<<std::endl;
    }
}gLogger;
void ZxfNet::build(){
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();

    IUffParser* parser = createUffParser();

    // parser->registerInput("per_image_standardization", DimsCHW(3, 100, 100), UffInputOrder::kNCHW);
    // parser->registerOutput("softmax_linear/softmax_linear_1");
    // parser->parse("zxf.uff", *network, nvinfer1::DataType::kFLOAT);

    parser->registerInput("x_placeholder", DimsCHW(1, 10, 10), UffInputOrder::kNHWC);
    parser->registerOutput("sum_fin_op");
    parser->parse("/home/nvidia/Desktop/test2x.uff", *network, nvinfer1::DataType::kFLOAT);
    //[E] [TRT] UffParser: Unsupported number of graph 0

    builder->setMaxBatchSize(1);
    IBuilderConfig* config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1 << 20);
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);//[E][TRT]Network must have at least one output


    IExecutionContext *context = engine->createExecutionContext();


    
//    std::cout<<engine->getNbBindings()<<std::endl;
//    std::cout<<66<<std::endl;
//    std::cout<<engine->getBindingName(0)<<std::endl;
//    std::cout<<engine->getBindingName(1)<<std::endl;
    int inputIndex = engine->getBindingIndex("x_placeholder");
    int outputIndex = engine->getBindingIndex("sum_fin_op");

    float * input_tensor;
  cudaHostAlloc((void**)&input_tensor, 10*10 * sizeof(float), cudaHostAllocMapped);
//input data init
  for (int i = 0; i < 10*10; i++)
  {
    input_tensor[i] = (float)1.5;
  }
  //out  = input +1+2+3=1.5+1+2+3=10.5
     // allocate memory on host / device for input / output
    float *output;
    float *inputDevice;
    float *outputDevice;
    size_t inputSize = 10 * 10 * sizeof(float);//输入输出的大小
    cudaHostAlloc((void**)&output, inputSize, cudaHostAllocMapped);//分配输出的内存

    cudaMalloc((void**)&inputDevice, inputSize);
    cudaMemcpy(inputDevice, input_tensor, inputSize, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&outputDevice, inputSize);

    void* buffers[2];
    // void* inputbuffer = nullptr;
    // void* outputbuffer = nullptr;
    buffers[inputIndex] = inputDevice;
    buffers[outputIndex] = outputDevice;
    std::cout<<outputIndex<<std::endl;
    int batchsize = 1;
    cudaStream_t stream = nullptr;
    context->execute(1, (void**)buffers);
    // context->enqueue(batchsize, buffers, stream, nullptr);
    cudaMemcpy(output, outputDevice, 10*10 * sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0;i<10;++i){
        for(int j = 0;j<10;j++){
            std::cout<<*output<<" ";
            output++;
        }
        std::cout<<std::endl;
    }
    std::cout<<"run success";
}
