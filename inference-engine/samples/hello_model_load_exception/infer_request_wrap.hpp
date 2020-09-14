// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#pragma once

#include <memory>
#include <map>
#include <string>
#include <chrono>
#include <random>

#include "inference_engine.hpp"

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::nanoseconds ns;

class InferReqWrap {
public:
    using Ptr = std::shared_ptr<InferReqWrap>;

    explicit InferReqWrap(InferenceEngine::ExecutableNetwork& net, std::string input_name) : _request(net.CreateInferRequest()),
                                                    _input_name(input_name) { }

    void startAsync() {
        auto blob = createRandomBlob();
        _request.SetBlob(_input_name, blob);
        _request.StartAsync();
    }

    void infer() {
        auto blob = createRandomBlob();
        _request.SetBlob(_input_name, blob);
        _request.Infer();
    }

    void wait() {
        InferenceEngine::StatusCode code = _request.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
        if (code != InferenceEngine::StatusCode::OK) {
            throw std::logic_error("Wait");
        }
    }

    InferenceEngine::Blob::Ptr getBlob(const std::string &name) {
        return _request.GetBlob(name);
    }

    InferenceEngine::Blob::Ptr createRandomBlob() {
        cv::Mat frame(224, 224, CV_8UC3);
        cv::randu(frame, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

        size_t channels = frame.channels();
        size_t height = frame.size().height;
        size_t width = frame.size().width;

        size_t strideH = frame.step.buf[0];
        size_t strideW = frame.step.buf[1];

        bool is_dense =
                strideW == channels &&
                strideH == channels * width;

        if (!is_dense) THROW_IE_EXCEPTION
                    << "Doesn't support conversion from not dense cv::Mat";

        InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::U8,
                                        {1, channels, height, width},
                                        InferenceEngine::Layout::NHWC);

        InferenceEngine::Blob::Ptr image_blob = InferenceEngine::make_shared_blob<uint8_t>(tDesc, frame.data);

        std::random_device rd;
        std::mt19937 rand_generator(rd());
        std::uniform_real_distribution<> dis(0.05, 0.5);

        double ratio_h = dis(rand_generator);
        double ratio_w = dis(rand_generator);
        InferenceEngine::ROI crop_roi({0, (size_t)(width * ratio_w), (size_t)(height * ratio_h),
                    (size_t)(width * (1 - ratio_w)), (size_t)(height * (1 - ratio_h))});
        image_blob = make_shared_blob(image_blob, crop_roi);
        return image_blob;
}

private:
    InferenceEngine::InferRequest _request;
    std::string _input_name;
};