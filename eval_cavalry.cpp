#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>

// Read the manifest file

// Generate a template for each of the binary images

// The rest is basically the same, great score distro and det curves
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
        imageInfo.templ = std::move(templ);
        imageInfoVec.emplace_back(std::move(imageInfo));
    }


}
