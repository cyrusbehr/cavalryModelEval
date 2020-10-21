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
    const unsigned int numTemplates = 30;
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

    int prevIdentity = -1;
    int counter = 0;

    while(std::getline(infile, line)) {
        offset++;
        if(offset < 10000) {
            continue;
        }

        std::istringstream tokenStream(line);
        std::vector<std::string> words;

        while(std::getline(tokenStream, token, ',')) {
            words.push_back(token);
        }

        std::cout << numProcessed << "/" << numTemplates << std::endl;
        if (numProcessed++ >= numTemplates) {
            break;
        }

        const auto path = words[1].substr(words[1].find_last_of('/'));
        const auto imagePath = parentDir + "data" + path;

        // Images are already cropped and aligned
        auto imgBGR = cv::imread(imagePath);
        cv::Mat imgRGB;
        cv::cvtColor(imgBGR, imgRGB, cv::COLOR_BGR2RGB);

        // Ensure the image size is correct
        if (imgRGB.cols != 112 || imgRGB.rows != 112) {
            std::cout << "Image at path: " << imagePath << " has incorrect size: " << imgRGB.rows << " " << imgRGB.cols << std::endl;
            continue;
        }

        Trueface::Faceprint faceprint;
        auto retCode = sdk.getFaceFeatureVector(imgRGB.data, faceprint);
        if (retCode != Trueface::ErrorCode::NO_ERROR) {
            continue;
        }

        ImageInfo imageInfo;
        imageInfo.faceprint = std::move(faceprint);
        imageInfo.identity = stoi(words[0]);
        collection.emplace_back(std::move(imageInfo));

        counter++;

        if (prevIdentity != imageInfo.identity) {
            prevIdentity = imageInfo.identity;
            counter = 0;
        }
        // Save all the images and save the descriptor to a manifest file

        std::string filename = words[0] + "_" + std::to_string(counter) + ".bin";
        std::string writePath = "../binaryImages/" + filename;
        std::ofstream outfile (writePath, std::ofstream::binary);

        outfile.write(reinterpret_cast<const char*>(imgRGB.data), 112 * 112 * 3);
        outfile.close();
    }

    return 0;
}
