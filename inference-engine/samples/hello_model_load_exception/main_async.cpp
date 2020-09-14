//  Copyright (C) 2018-2019 Intel Corporation
//  SPDX-License-Identifier: Apache-2.0
//

#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>

#include "infer_request_wrap.hpp"

constexpr size_t batchSize = 1;
uint32_t nireq = 4;

using namespace InferenceEngine;

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cout << "Usage : ./hello_classification <path_to_model> "
                 "<num_of_models> <nireq>"
              << std::endl;
    return EXIT_FAILURE;
  }
  const std::string input_model{argv[1]};
  const uint32_t model_num = std::stoul(argv[2]);
  if (argc == 4)
    nireq = std::stoul(argv[3]);

  Core ie;

  std::cout << "Reading networks..." << std::endl;
  std::vector<ExecutableNetwork> executable_networks;
  static std::string input_name;

  for (size_t i = 0; i < model_num; i++) {
    CNNNetwork network = ie.ReadNetwork(input_model);

    InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;
    input_name = network.getInputsInfo().begin()->first;

    input_info->getPreProcess().setResizeAlgorithm(
        ResizeAlgorithm::RESIZE_BILINEAR);
    input_info->getPreProcess().setColorFormat(ColorFormat::BGR);

    input_info->setLayout(Layout::NCHW);
    input_info->setPrecision(Precision::U8);

    DataPtr output_info = network.getOutputsInfo().begin()->second;
    std::string output_name = network.getOutputsInfo().begin()->first;

    output_info->setPrecision(Precision::FP32);

    executable_networks.push_back(ie.LoadNetwork(network, "CPU"));
  }

  std::cout << "Initializing IE requests..." << std::endl;
  std::vector<InferReqWrap::Ptr> inferRequests;
  inferRequests.reserve(nireq);

  for (size_t i = 0; i < nireq; i++) {
    inferRequests.push_back(std::make_shared<InferReqWrap>(
        executable_networks[i % model_num], input_name));
  }

  std::cout << "Start inference..." << std::endl;
  inferRequests[0]->startAsync();
  inferRequests[0]->wait();

  long long currentInference = 0LL;
  long long previousInference = 1LL - nireq;

  using namespace std::chrono;
  seconds working_time{30};
  auto start = high_resolution_clock::now();

  while (duration_cast<seconds>(high_resolution_clock::now() - start) <
         working_time) {
    // start new inference
    inferRequests[currentInference]->startAsync();

    // wait the latest inference execution if exists
    if (previousInference >= 0) {
      inferRequests[previousInference]->wait();
    }

    currentInference++;
    if (currentInference >= nireq) {
      currentInference = 0;
    }

    previousInference++;
    if (previousInference >= nireq) {
      previousInference = 0;
    }
  }

  // wait the latest inference executions
  for (size_t notCompletedIndex = 0ULL; notCompletedIndex < (nireq - 1);
       ++notCompletedIndex) {
    if (previousInference >= 0) {
      inferRequests[previousInference]->wait();
    }

    previousInference++;
    if (previousInference >= nireq) {
      previousInference = 0LL;
    }
  }

  return 0;
}
