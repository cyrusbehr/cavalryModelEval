#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <cmath>
#include <algorithm>


float dotProduct(const std::vector<float>& v1, const std::vector<float>& v2) {
    if (v1.size() != v2.size() || v1.empty()) {
        throw std::runtime_error("Vector size is incorrect");
    }

    float dotProduct = 0.f;
    for (size_t i = 0; i < v1.size(); ++i) {
        dotProduct += v1[i] * v2[i];
    }

    return dotProduct;
}

void normalizeVector(std::vector<float>& v) {
    float magnitude = std::sqrt(dotProduct(v, v));
    for (float & j : v) {
        j = j / magnitude;
    }
}

struct ImageInfo {
    int identity;
    std::vector<float> templ;
};

int main(int argc, char **argv) {
    if (argc != 3) {
        std::string errorMsg = "Error, must provide two arguments. arg1: path to directory containing binaryImage and manigest (including /)."
                               " arg2: full path to calvary model, including model name.";
        throw std::runtime_error(errorMsg);
    }

    auto imageDirectoryPath = std::string(argv[1]);
    auto calvaryModelPath = std::string(argv[2]);

    // Read the manifest file

    auto filename = imageDirectoryPath + "manifest.txt";
    std::ifstream myFile(filename);
    // Make sure the file is open
    if(!myFile.is_open()) {
        throw std::runtime_error("Could not open file");
    }

    std::vector<std::string> imageList;
    std::vector<ImageInfo> imageInfoVec;
    std::vector<float> genuineScores;
    std::vector<float> impostorScores;

    std::string a;
    while (myFile >> a)
    {
        imageList.emplace_back(std::move(a));
    }

    if (imageList.empty()) {
        throw std::runtime_error("Image list is empty!");
    }

    system("modprobe cavalry; cavalry_load -f /lib/firmware/cavalry.bin -r");

    std::cout << "------------------------------------------------" << std::endl;

    // Generate a template for each of the images
    int iteration = 0;
    for (const auto& image: imageList) {
        std::cout << ++iteration << "/" << imageList.size() << std::endl;
        std::cout << image << std::endl;
        auto imagePath = imageDirectoryPath + image;

        std::string command = "test_nnctrl -b " + calvaryModelPath + " --in data=" + imagePath + " --out fc1=template.bin -e";
        std::cout << command << std::endl;
        int ret = system(command.c_str());

        if (ret!= 0) {
            std::cout << "Unable to execute command" << std::endl;
            continue;
        }

        // Read the template
        std::ifstream templFile("template.bin", std::ifstream::binary | std::ios::ate);
        if (!templFile.good()) {
            std::cout << "Unable to read template.bin" << std::endl;
            continue;
        }

        std::streamsize size = templFile.tellg();
        if (size != 2048) {
            std::cout << "Unexpected file size: " << size << std::endl;
        }

        templFile.seekg(0, std::ios::beg);
        std::vector<float> templ (size / sizeof(float));
        if (!templFile.read(reinterpret_cast<char*>(templ.data()), size)) {
            std::cout << "Unable to read file into buffer" << std::endl;
            continue;
        }

        // Get the image identity from the filename
        auto identity = image.substr(0, image.find('_'));
        std::stringstream ss(identity);

        ImageInfo imageInfo;
        ss >> imageInfo.identity;
        std::cout << imageInfo.identity << std::endl;

        // Normalize the vector
        normalizeVector(templ);
        imageInfo.templ = std::move(templ);
        imageInfoVec.emplace_back(std::move(imageInfo));
    }

    const std::string scoresFilename = "mugshots_scores_full_model_cavalry_" + std::to_string(imageList.size()) + "_templates.csv";
    std::ofstream scoresOut(scoresFilename);

    for (size_t i = 0; i < imageInfoVec.size() - 1; ++i) {
        std::cout << "Running comparison: " << i << "/" << imageInfoVec.size() << std::endl;
        for (size_t j = i + 1; j < imageInfoVec.size(); ++j) {
            float similarity = dotProduct(imageInfoVec[i].templ, imageInfoVec[j].templ);

            int compType = 0;
            if (imageInfoVec[i].identity == imageInfoVec[j].identity) {
                compType = 1;
                genuineScores.push_back(similarity);
            } else {
                impostorScores.push_back(similarity);
                if (similarity > 0.5) {
                    std::cout << similarity << std::endl;
                    std::cout << imageInfoVec[i].identity << std::endl;
                    std::cout << imageInfoVec[j].identity << std::endl;
                    std::cout << std::endl;
                }
            }

            scoresOut << compType << ',' << similarity << '\n';
        }
    }

    std::cout << "Sorting vectors" << std::endl;
    std::sort(impostorScores.begin(), impostorScores.end());
    std::sort(genuineScores.begin(), genuineScores.end());

    std::vector<float> FPR;
    std::vector<float> FNR;

    double threshold = 0.0;
    const double increment = 0.001;
    size_t i = 0;

    std::cout << "Computing FPR" << std::endl;
    while ( i < impostorScores.size()) {
        if (impostorScores[i] > threshold) {
            threshold += increment;
            FPR.push_back((impostorScores.size() - static_cast<double>(i)) / impostorScores.size());
        } else {
            ++i;
        }
    }

    while(threshold < 1 + increment) {
        FPR.push_back(0);
        threshold += increment;
    }

    threshold = 0.0;
    i = 0;
    std::cout << "Computing FNR" << std::endl;
    while ( i < genuineScores.size()) {
        if (genuineScores[i] > threshold) {
            threshold += increment;
            FNR.push_back((static_cast<double>(i)) / genuineScores.size());
        } else {
            ++i;
        }
    }

    while(threshold < 1 + increment) {
        FNR.push_back(0);
        threshold += increment;
    }

    std::cout << FNR.size() << std::endl;
    std::cout << FPR.size() << std::endl;


    std::string FPRfilename = "mugshot_FPR_cavalry.csv";
    std::ofstream FPRfile(FPRfilename);

    std::string FNRfilename = "mugshot_FNR_cavalry.csv";
    std::ofstream FNRfile(FNRfilename);

    for (float idx : FPR) {
        FPRfile << idx << '\n';
    }

    for (float idx : FNR) {
        FNRfile << idx << '\n';
    }


}
