#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include "tf_sdk.h"

struct ImageInfo {
    Trueface::Faceprint faceprint;
    int identity;
};



int main() {
    // --------------- Configuration parameters ---------------------
    const std::string parentDir = "/home/cyrus/work/data/mugshots/";
    const std::string descriptorFilename = "unique.txt";
    const unsigned int numTemplates = 1000;
    // --------------- Configuration parameters ---------------------

    Trueface::ConfigurationOptions options;
    options.frModel = Trueface::FacialRecognitionModel::FULL;
    options.frVectorCompression = false;
    options.smallestFaceHeight = 40;
    Trueface::SDK sdk(options);

    std::vector<ImageInfo> collection;
    std::vector<float> genuineScores;
    std::vector<float> impostorScores;

    auto ret = sdk.setLicense("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbW90aW9uIjpudWxsLCJmciI6bnVsbCwiYnlwYXNzX2dwdV91dWlkIjp0cnVlLCJwYWNrYWdlX2lkIjpudWxsLCJleHBpcnlfZGF0ZSI6IjIwMjAtMTItMzEiLCJncHVfdXVpZCI6W10sInRocmVhdF9kZXRlY3Rpb24iOm51bGwsIm1hY2hpbmVzIjoxLCJhbHByIjpudWxsLCJuYW1lIjoiQ3lydXMiLCJ0a2V5IjoibmV3IiwiZXhwaXJ5X3RpbWVfc3RhbXAiOjE2MDkzNzI4MDAuMCwiYXR0cmlidXRlcyI6bnVsbCwidHlwZSI6Im9mZmxpbmUiLCJlbWFpbCI6ImN5cnVzQHRydWVmYWNlLmFpIn0.oyZZSi_HICgbM69Y-0Fq5rcPrAR7JKeUkQ9YNf2PQsM");
    if (!ret) {
        std::cout << "License not valid" << std::endl;
        return -1;
    }

    const auto descriptorFilepath = parentDir + descriptorFilename;

    std::ifstream infile(descriptorFilepath);
    if (!infile.good()) {
        std::cout << "Unable to open file: " << descriptorFilepath << std::endl;
        return -1;
    }

    const std::string scoresFilename = "mugshots_scores_full_model_sdk_" + std::to_string(numTemplates) + "_templates.csv";
    std::ofstream scoresOut(scoresFilename);

    std::string line;
    std::string token;
    int numProcessed = 0;
    int offset = 0;

    return 0;
}
