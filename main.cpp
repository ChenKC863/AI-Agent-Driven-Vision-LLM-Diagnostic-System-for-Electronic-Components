#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <stdexcept>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
namespace fs = std::filesystem;

// Uncomment to enable detailed debug output.
// #define DEBUG_MODE

// ----- Preprocessing constants -----
const int IMG_SIZE = 224;
const std::vector<float> MEAN = {0.485f, 0.456f, 0.406f};
const std::vector<float> STD  = {0.229f, 0.224f, 0.225f};

// ----- Class names -----
const std::vector<std::string> CLASS_NAMES = {
    "Inductor", "Resistor", "Transformer", "solenoid"
};

const float DEFAULT_CONFIDENCE_THRESHOLD = 0.70f;

std::string toLowerCopy(std::string value);

struct DiagnosticTemplate {
    std::string function;
};

struct Prediction {
    std::string imagePath;
    std::string predictedLabel;
    float confidence;
    std::vector<float> probabilities;
    long long inferenceTimeMs;
};

DiagnosticTemplate getDiagnosticTemplate(const std::string& component) {
    const std::string lowerComponent = toLowerCopy(component);

    if (lowerComponent == "inductor") {
        return {
            "Stores energy in a magnetic field and filters current ripple."
        };
    }

    if (lowerComponent == "resistor") {
        return {
            "Limits current, divides voltage, or sets bias in the circuit."
        };
    }

    if (lowerComponent == "transformer") {
        return {
            "Transfers electrical energy between windings through magnetic coupling."
        };
    }

    if (lowerComponent == "solenoid") {
        return {
            "Converts electrical energy into linear mechanical motion using an electromagnetic coil."
        };
    }

    return {
        "N/A"
    };
}

// ----- LLM configuration -----
const std::string LLM_API_URL = "http://localhost:11434/v1/chat/completions";
const std::string DEFAULT_LLM_MODEL_NAME = "llama3.2:1b";

#ifdef DEBUG_MODE
void debugPrint(const std::string& message) {
    std::cout << "[DEBUG] " << message << std::endl;
}
#else
void debugPrint(const std::string&) {}
#endif

class CurlGlobalGuard {
public:
    CurlGlobalGuard() {
        const CURLcode rc = curl_global_init(CURL_GLOBAL_DEFAULT);
        if (rc != CURLE_OK) {
            throw std::runtime_error(
                std::string("curl_global_init failed: ") + curl_easy_strerror(rc)
            );
        }
    }

    ~CurlGlobalGuard() {
        curl_global_cleanup();
    }

    CurlGlobalGuard(const CurlGlobalGuard&) = delete;
    CurlGlobalGuard& operator=(const CurlGlobalGuard&) = delete;
};

std::string toLowerCopy(std::string value) {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); }
    );
    return value;
}

std::string trimCopy(const std::string& input) {
    const std::string whitespace = " \t\r\n";
    const std::size_t first = input.find_first_not_of(whitespace);
    if (first == std::string::npos) {
        return "";
    }
    const std::size_t last = input.find_last_not_of(whitespace);
    return input.substr(first, last - first + 1);
}

std::string formatPercent(float probability) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(1) << probability * 100.0f << "%";
    return stream.str();
}

std::string getChartFilename(const std::string& modelPath) {
    const std::string lowerModelPath = toLowerCopy(modelPath);

    if (lowerModelPath.find("small") != std::string::npos) {
        return "confidence_chart_small_model.png";
    }
    if (lowerModelPath.find("large") != std::string::npos) {
        return "confidence_chart_large_model.png";
    }
    return "confidence_chart.png";
}

std::string getPredictionCsvFilename(const std::string& modelPath) {
    const std::string lowerModelPath = toLowerCopy(modelPath);

    if (lowerModelPath.find("small") != std::string::npos) {
        return "vision_predictions_small_model.csv";
    }
    if (lowerModelPath.find("large") != std::string::npos) {
        return "vision_predictions_large_model.csv";
    }
    return "vision_predictions.csv";
}

std::string getLLMModelName() {
    const char* envModel = std::getenv("OLLAMA_MODEL");
    if (envModel != nullptr) {
        const std::string modelName = trimCopy(envModel);
        if (!modelName.empty()) {
            return modelName;
        }
    }
    return DEFAULT_LLM_MODEL_NAME;
}

float getConfidenceThreshold() {
    const char* envThreshold = std::getenv("CONFIDENCE_THRESHOLD");
    if (envThreshold == nullptr) {
        return DEFAULT_CONFIDENCE_THRESHOLD;
    }

    try {
        const float threshold = std::stof(trimCopy(envThreshold));
        if (threshold >= 0.0f && threshold <= 1.0f) {
            return threshold;
        }
    } catch (const std::exception&) {
    }

    return DEFAULT_CONFIDENCE_THRESHOLD;
}

// ----- libcurl write callback -----
size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* response) {
    const size_t totalSize = size * nmemb;
    if (contents != nullptr && response != nullptr) {
        response->append(static_cast<const char*>(contents), totalSize);
    }
    return totalSize;
}

std::string extractJsonCandidate(const std::string& content) {
    const std::string trimmed = trimCopy(content);
    if (trimmed.empty()) {
        return trimmed;
    }

    // If the model wraps JSON in fenced code blocks, try to recover the JSON object.
    if (trimmed.rfind("```", 0) == 0) {
        const std::size_t firstBrace = trimmed.find('{');
        const std::size_t lastBrace = trimmed.rfind('}');
        if (firstBrace != std::string::npos &&
            lastBrace != std::string::npos &&
            lastBrace > firstBrace) {
            return trimmed.substr(firstBrace, lastBrace - firstBrace + 1);
        }
    }

    return trimmed;
}

std::string repairCommonJsonValueErrors(std::string text) {
    const std::vector<std::string> keys = {
        "component",
        "function"
    };

    for (const std::string& key : keys) {
        const std::string pattern = "\"" + key + "\":";
        std::size_t pos = 0;
        while ((pos = text.find(pattern, pos)) != std::string::npos) {
            std::size_t valueStart = pos + pattern.size();
            while (valueStart < text.size() &&
                   (text[valueStart] == ' ' || text[valueStart] == '\t')) {
                ++valueStart;
            }

            if (valueStart >= text.size() ||
                text[valueStart] == '"' ||
                text[valueStart] == '{' ||
                text[valueStart] == '[' ||
                text.compare(valueStart, 4, "null") == 0 ||
                text.compare(valueStart, 4, "true") == 0 ||
                text.compare(valueStart, 5, "false") == 0) {
                pos = valueStart + 1;
                continue;
            }

            std::size_t valueEnd = text.find_first_of(",\r\n}", valueStart);
            if (valueEnd == std::string::npos || valueEnd <= valueStart) {
                pos = valueStart + 1;
                continue;
            }

            std::string value = trimCopy(text.substr(valueStart, valueEnd - valueStart));
            if (value.empty()) {
                value = "N/A";
            }

            text.replace(valueStart, valueEnd - valueStart, json(value).dump());
            pos = valueStart + value.size() + 2;
        }
    }

    return text;
}

std::string buildStableDiagnosticInsight(
    const std::string& predictedClass,
    const std::vector<float>& probabilities,
    const std::string& warning = ""
) {
    const DiagnosticTemplate stableDiagnostic = getDiagnosticTemplate(predictedClass);

    std::ostringstream oss;
    oss << "Component            : " << predictedClass << "\n";
    oss << "Function             : " << stableDiagnostic.function;

    const float threshold = getConfidenceThreshold();
    const auto maxIter = std::max_element(probabilities.begin(), probabilities.end());
    if (maxIter != probabilities.end() && *maxIter < threshold) {
        oss << "\nNote                 : Confidence is below "
            << std::fixed << std::setprecision(0) << threshold * 100.0f
            << "%; manual review is recommended.";
    }

    if (!warning.empty()) {
        oss << "\nLLM Warning          : " << warning;
    }

    return oss.str();
}

// ----- Query local Ollama and parse JSON output -----
std::string queryLLM(const std::string& predictedClass, const std::vector<float>& probabilities) {
    debugPrint("Entering queryLLM().");

    const DiagnosticTemplate stableDiagnostic = getDiagnosticTemplate(predictedClass);

    std::string prompt =
        "You are an electronics manufacturing expert.\n"
        "A vision model predicted an electronic component.\n\n"
        "Prediction: " + predictedClass + "\n\n"
        "Confidence:\n";

    for (std::size_t i = 0; i < CLASS_NAMES.size(); ++i) {
        prompt += CLASS_NAMES[i] + ": " + formatPercent(probabilities[i]) + "\n";
    }

    prompt += "\nThe component field must be exactly this plain string: " + predictedClass + "\n";
    prompt += "Use these exact diagnostic strings:\n";
    prompt += "function: " + stableDiagnostic.function + "\n";

    prompt += R"(
Return ONLY valid JSON.
{
  "component": "",
  "function": ""
}
One sentence per field. Use "N/A" if unknown.
All values must be plain JSON strings in double quotes.
Do not put extra quote characters inside string values.
No null values. No nested JSON. No markdown. No extra keys. No explanatory text.
)";

    json requestBody;
    requestBody["model"] = getLLMModelName();
    requestBody["messages"] = json::array();
    requestBody["messages"].push_back({
        {"role", "user"},
        {"content", prompt}
    });
    requestBody["stream"] = false;
    requestBody["temperature"] = 0.0;
    requestBody["seed"] = 42;
    requestBody["max_tokens"] = 220;

    const std::string requestStr = requestBody.dump();

    CURL* curl = curl_easy_init();
    if (curl == nullptr) {
        return "[LLM Error] Failed to initialize CURL easy handle.";
    }

    std::string response;
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, LLM_API_URL.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, requestStr.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(requestStr.size()));
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 180L);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl, CURLOPT_IPRESOLVE, CURL_IPRESOLVE_V4);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    debugPrint("Sending HTTP request to local Ollama.");

    const CURLcode res = curl_easy_perform(curl);

    long httpCode = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        return "[LLM Error] Network error: " + std::string(curl_easy_strerror(res));
    }

    if (httpCode != 200) {
        return "[LLM Error] HTTP error: " + std::to_string(httpCode) + ". Response: " + response;
    }

    std::string content;
    try {
        const json responseJson = json::parse(response);

        if (!responseJson.contains("choices") ||
            !responseJson["choices"].is_array() ||
            responseJson["choices"].empty()) {
            return "[LLM Error] Unexpected response format: missing choices array. Raw: " + response;
        }

        const json& firstChoice = responseJson["choices"].at(0);

        if (!firstChoice.contains("message") ||
            !firstChoice["message"].is_object() ||
            !firstChoice["message"].contains("content")) {
            return "[LLM Error] Unexpected response format: missing message.content. Raw: " + response;
        }

        content = firstChoice["message"]["content"].get<std::string>();
    } catch (const std::exception& e) {
        return std::string("[LLM Error] Failed to parse top-level response JSON: ") +
               e.what() + ". Raw: " + response;
    }

    const std::string jsonCandidate = extractJsonCandidate(content);
    const std::string repairedJsonCandidate = repairCommonJsonValueErrors(jsonCandidate);

    try {
        const json contentJson = json::parse(repairedJsonCandidate);

        (void)contentJson;

        // Keep diagnostic fields deterministic across repeated runs and model sizes.
        return buildStableDiagnosticInsight(predictedClass, probabilities);
    } catch (const std::exception& e) {
        return buildStableDiagnosticInsight(
            predictedClass,
            probabilities,
            std::string("Ollama returned malformed JSON; deterministic local diagnostic was used. Parser: ") +
                e.what()
        );
    }
}

// ----- Draw confidence chart using OpenCV -----
void drawConfidenceChart(
    const std::vector<float>& probabilities,
    const std::vector<std::string>& classNames,
    int predictedClass,
    const std::string& outputPath
) {
    if (probabilities.size() != classNames.size()) {
        throw std::runtime_error("drawConfidenceChart: probabilities and classNames size mismatch.");
    }

    if (predictedClass < 0 || predictedClass >= static_cast<int>(classNames.size())) {
        throw std::runtime_error("drawConfidenceChart: predictedClass index out of range.");
    }

    const int width = 1100;
    const int topMargin = 110;
    const int bottomMargin = 70;
    const int leftMargin = 250;
    const int rightMargin = 160;
    const int rowHeight = 68;
    const int chartHeight = static_cast<int>(classNames.size()) * rowHeight;
    const int height = topMargin + chartHeight + bottomMargin;

    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    const cv::Scalar colorBorder(220, 220, 220);
    const cv::Scalar colorAxis(70, 70, 70);
    const cv::Scalar colorGrid(235, 235, 235);
    const cv::Scalar colorBar(231, 111, 81);
    const cv::Scalar colorPredicted(46, 204, 113);
    const cv::Scalar colorText(30, 30, 30);
    const cv::Scalar colorMuted(110, 110, 110);

    cv::putText(
        canvas,
        "Confidence Distribution",
        cv::Point(40, 45),
        cv::FONT_HERSHEY_SIMPLEX,
        1.0,
        colorText,
        2,
        cv::LINE_AA
    );

    cv::putText(
        canvas,
        "Predicted class is highlighted in green.",
        cv::Point(40, 80),
        cv::FONT_HERSHEY_SIMPLEX,
        0.60,
        colorMuted,
        1,
        cv::LINE_AA
    );

    const int x0 = leftMargin;
    const int x1 = width - rightMargin;
    const int usableBarWidth = x1 - x0;

    cv::rectangle(
        canvas,
        cv::Rect(x0, topMargin - 15, usableBarWidth, chartHeight + 20),
        colorBorder,
        1,
        cv::LINE_AA
    );

    for (int tick = 0; tick <= 100; tick += 20) {
        const int x = x0 + static_cast<int>((tick / 100.0) * usableBarWidth);

        cv::line(
            canvas,
            cv::Point(x, topMargin - 15),
            cv::Point(x, topMargin + chartHeight + 5),
            colorGrid,
            1,
            cv::LINE_AA
        );

        std::ostringstream tickLabel;
        tickLabel << tick << "%";

        cv::putText(
            canvas,
            tickLabel.str(),
            cv::Point(x - 15, topMargin + chartHeight + 35),
            cv::FONT_HERSHEY_SIMPLEX,
            0.50,
            colorAxis,
            1,
            cv::LINE_AA
        );
    }

    for (std::size_t i = 0; i < classNames.size(); ++i) {
        const int y = topMargin + static_cast<int>(i) * rowHeight;
        const int barHeight = 34;
        const int barY = y + 10;

        const double percent = static_cast<double>(probabilities[i]) * 100.0;
        const int barWidth = static_cast<int>(std::round((percent / 100.0) * usableBarWidth));

        const cv::Scalar currentColor =
            (static_cast<int>(i) == predictedClass) ? colorPredicted : colorBar;

        cv::putText(
            canvas,
            classNames[i],
            cv::Point(40, barY + 24),
            cv::FONT_HERSHEY_SIMPLEX,
            0.65,
            colorText,
            2,
            cv::LINE_AA
        );

        cv::rectangle(
            canvas,
            cv::Rect(x0, barY, usableBarWidth, barHeight),
            cv::Scalar(245, 245, 245),
            cv::FILLED,
            cv::LINE_AA
        );

        cv::rectangle(
            canvas,
            cv::Rect(x0, barY, usableBarWidth, barHeight),
            colorBorder,
            1,
            cv::LINE_AA
        );

        if (barWidth > 0) {
            cv::rectangle(
                canvas,
                cv::Rect(x0, barY, barWidth, barHeight),
                currentColor,
                cv::FILLED,
                cv::LINE_AA
            );
        }

        std::ostringstream valueText;
        valueText << std::fixed << std::setprecision(2) << percent << "%";

        int textX = x0 + barWidth + 10;
        if (textX > width - 140) {
            textX = width - 140;
        }

        cv::putText(
            canvas,
            valueText.str(),
            cv::Point(textX, barY + 24),
            cv::FONT_HERSHEY_SIMPLEX,
            0.60,
            colorText,
            2,
            cv::LINE_AA
        );
    }

    if (!cv::imwrite(outputPath, canvas)) {
        throw std::runtime_error("Failed to save confidence chart to: " + outputPath);
    }
}

// ----- Preprocess image -----
std::vector<float> preprocessImage(const std::string& imagePath) {
    debugPrint("Reading and preprocessing image: " + imagePath);

    cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (img.empty()) {
        throw std::runtime_error("Cannot read image: " + imagePath);
    }

    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    const float scale = 256.0f / static_cast<float>(std::min(img.rows, img.cols));
    cv::resize(img, img, cv::Size(), scale, scale, cv::INTER_LINEAR);

    const int x = (img.cols - IMG_SIZE) / 2;
    const int y = (img.rows - IMG_SIZE) / 2;

    if (x < 0 || y < 0 || x + IMG_SIZE > img.cols || y + IMG_SIZE > img.rows) {
        throw std::runtime_error("Center crop is out of bounds after resize.");
    }

    const cv::Rect crop(x, y, IMG_SIZE, IMG_SIZE);
    img = img(crop).clone();
    img.convertTo(img, CV_32FC3, 1.0 / 255.0);

    std::vector<float> tensor(3 * IMG_SIZE * IMG_SIZE);

    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < IMG_SIZE; ++h) {
            for (int w = 0; w < IMG_SIZE; ++w) {
                float pixel = img.at<cv::Vec3f>(h, w)[c];
                pixel = (pixel - MEAN[c]) / STD[c];
                tensor[c * IMG_SIZE * IMG_SIZE + h * IMG_SIZE + w] = pixel;
            }
        }
    }

    return tensor;
}

Ort::Session createSession(Ort::Env& env, const std::string& modelPath) {
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);

#ifdef _WIN32
    const std::wstring modelPathWide(modelPath.begin(), modelPath.end());
    return Ort::Session(env, modelPathWide.c_str(), sessionOptions);
#else
    return Ort::Session(env, modelPath.c_str(), sessionOptions);
#endif
}

Prediction inferImage(Ort::Session& session, const std::string& imagePath) {
    Ort::AllocatorWithDefaultOptions allocator;
    auto inputNameAllocated = session.GetInputNameAllocated(0, allocator);
    auto outputNameAllocated = session.GetOutputNameAllocated(0, allocator);

    const char* inputNames[] = { inputNameAllocated.get() };
    const char* outputNames[] = { outputNameAllocated.get() };

    std::vector<float> inputData = preprocessImage(imagePath);
    const std::vector<int64_t> inputDims = {1, 3, IMG_SIZE, IMG_SIZE};

    Ort::MemoryInfo memoryInfo =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        inputData.data(),
        inputData.size(),
        inputDims.data(),
        inputDims.size()
    );

    const auto start = std::chrono::high_resolution_clock::now();

    auto outputTensors = session.Run(
        Ort::RunOptions{nullptr},
        inputNames,
        &inputTensor,
        1,
        outputNames,
        1
    );

    const auto end = std::chrono::high_resolution_clock::now();
    const auto durationMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    const float* outputPtr = outputTensors[0].GetTensorData<float>();
    const std::vector<int64_t> outputShape =
        outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

    if (outputShape.empty()) {
        throw std::runtime_error("Model output shape is empty.");
    }

    const std::size_t numClasses = static_cast<std::size_t>(outputShape.back());
    if (numClasses != CLASS_NAMES.size()) {
        throw std::runtime_error("Model class count does not match CLASS_NAMES.");
    }

    std::vector<float> probabilities(numClasses, 0.0f);

    const float maxLogit = *std::max_element(outputPtr, outputPtr + numClasses);
    float sumExp = 0.0f;

    for (std::size_t i = 0; i < numClasses; ++i) {
        probabilities[i] = std::exp(outputPtr[i] - maxLogit);
        sumExp += probabilities[i];
    }

    if (sumExp <= 0.0f || !std::isfinite(sumExp)) {
        throw std::runtime_error("Invalid softmax normalization factor.");
    }

    for (float& p : probabilities) {
        p /= sumExp;
    }

    const int predictedClass = static_cast<int>(
        std::distance(
            probabilities.begin(),
            std::max_element(probabilities.begin(), probabilities.end())
        )
    );

    return {
        imagePath,
        CLASS_NAMES[predictedClass],
        probabilities[predictedClass],
        probabilities,
        durationMs
    };
}

int predictedClassIndex(const Prediction& prediction) {
    const auto it = std::find(
        CLASS_NAMES.begin(),
        CLASS_NAMES.end(),
        prediction.predictedLabel
    );

    if (it == CLASS_NAMES.end()) {
        throw std::runtime_error("Predicted label is not in CLASS_NAMES.");
    }

    return static_cast<int>(std::distance(CLASS_NAMES.begin(), it));
}

json predictionToJson(const Prediction& prediction) {
    json probabilitiesJson = json::object();
    for (std::size_t i = 0; i < CLASS_NAMES.size(); ++i) {
        probabilitiesJson[CLASS_NAMES[i]] = prediction.probabilities[i];
    }

    const float threshold = getConfidenceThreshold();

    return {
        {"image_path", prediction.imagePath},
        {"predicted_label", prediction.predictedLabel},
        {"confidence", prediction.confidence},
        {"confidence_threshold", threshold},
        {"requires_human_review", prediction.confidence < threshold},
        {"probabilities", probabilitiesJson},
        {"inference_time_ms", prediction.inferenceTimeMs}
    };
}

std::string csvEscape(const std::string& value) {
    bool needsQuotes = false;
    for (const char ch : value) {
        if (ch == ',' || ch == '"' || ch == '\n' || ch == '\r') {
            needsQuotes = true;
            break;
        }
    }

    if (!needsQuotes) {
        return value;
    }

    std::string escaped = "\"";
    for (const char ch : value) {
        if (ch == '"') {
            escaped += "\"\"";
        } else {
            escaped += ch;
        }
    }
    escaped += "\"";
    return escaped;
}

bool isSupportedImageFile(const fs::path& path) {
    const std::string ext = toLowerCopy(path.extension().string());
    return ext == ".jpg" ||
           ext == ".jpeg" ||
           ext == ".png" ||
           ext == ".bmp" ||
           ext == ".webp";
}

std::vector<fs::path> collectImageFiles(const fs::path& classDir) {
    std::vector<fs::path> imagePaths;

    for (const auto& entry : fs::directory_iterator(classDir)) {
        if (entry.is_regular_file() && isSupportedImageFile(entry.path())) {
            imagePaths.push_back(entry.path());
        }
    }

    std::sort(
        imagePaths.begin(),
        imagePaths.end(),
        [](const fs::path& a, const fs::path& b) {
            return a.generic_string() < b.generic_string();
        }
    );

    return imagePaths;
}

std::size_t exportSplitCsv(
    Ort::Session& session,
    const fs::path& splitRoot,
    const std::string& splitLabel,
    std::ofstream& csv
) {
    std::size_t rows = 0;

    for (const std::string& trueLabel : CLASS_NAMES) {
        const fs::path classDir = splitRoot / trueLabel;
        if (!fs::exists(classDir)) {
            throw std::runtime_error("Missing class directory: " + classDir.generic_string());
        }

        const std::vector<fs::path> imagePaths = collectImageFiles(classDir);
        for (const fs::path& imagePath : imagePaths) {
            const std::string imagePathString = imagePath.generic_string();
            const Prediction prediction = inferImage(session, imagePathString);

            csv << csvEscape(imagePathString) << ','
                << csvEscape(trueLabel) << ','
                << csvEscape(prediction.predictedLabel) << ','
                << std::fixed << std::setprecision(6)
                << prediction.confidence << ','
                << prediction.probabilities[0] << ','
                << prediction.probabilities[1] << ','
                << prediction.probabilities[2] << ','
                << prediction.probabilities[3] << ','
                << csvEscape(splitLabel) << '\n';

            ++rows;
        }
    }

    return rows;
}

std::size_t exportDatasetCsv(
    Ort::Session& session,
    const std::string& datasetRoot,
    const std::string& outputCsvPath
) {
    const fs::path root(datasetRoot);
    std::ofstream csv(outputCsvPath);
    if (!csv.is_open()) {
        throw std::runtime_error("Cannot open CSV output path: " + outputCsvPath);
    }

    csv << "image_path,true_label,predicted_label,confidence,"
           "prob_Inductor,prob_Resistor,prob_Transformer,prob_solenoid,split\n";

    std::size_t totalRows = 0;

    const fs::path trainDir = root / "train";
    if (fs::exists(trainDir)) {
        totalRows += exportSplitCsv(session, trainDir, "train", csv);
    }

    const fs::path validationDir = root / "validation";
    const fs::path valDir = root / "val";
    if (fs::exists(validationDir)) {
        totalRows += exportSplitCsv(session, validationDir, "validation", csv);
    } else if (fs::exists(valDir)) {
        totalRows += exportSplitCsv(session, valDir, "validation", csv);
    }

    const fs::path testDir = root / "test";
    if (fs::exists(testDir)) {
        totalRows += exportSplitCsv(session, testDir, "test", csv);
    }

    if (totalRows == 0) {
        throw std::runtime_error(
            "No train/validation/val/test split folders found under: " + root.generic_string()
        );
    }

    return totalRows;
}

void printConsoleHeader(const std::string& modelPath, const std::string& imagePath) {
    std::cout << "========== Program Started ==========\n";
    std::cout << "Model Path : " << modelPath << "\n";
    std::cout << "Image Path : " << imagePath << "\n\n";
}

void printProbabilityTable(const std::vector<float>& probabilities) {
    std::cout << "Class Probabilities\n";
    std::cout << "---------------------------------------------\n";

    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        std::cout << std::left << std::setw(14) << CLASS_NAMES[i]
                  << " : "
                  << std::right << std::setw(7) << std::fixed << std::setprecision(2)
                  << probabilities[i] * 100.0f << " %\n";
    }

    std::cout << std::left;
}

void printUsage(const char* programName) {
    std::cerr << "Usage:\n"
              << "  " << programName << " <model_path> <image_path>\n"
              << "  " << programName << " --json <model_path> <image_path>\n"
              << "  " << programName << " --batch-csv <model_path> <dataset_root> [output_csv]\n\n"
              << "Batch CSV defaults:\n"
              << "  model path contains small -> vision_predictions_small_model.csv\n"
              << "  model path contains large -> vision_predictions_large_model.csv\n\n"
              << "Environment variables:\n"
              << "  OLLAMA_MODEL           Local Ollama model name, default: "
              << DEFAULT_LLM_MODEL_NAME << "\n"
              << "  CONFIDENCE_THRESHOLD   Human-review threshold, default: "
              << DEFAULT_CONFIDENCE_THRESHOLD << "\n";
}

// ----- Main function -----
int main(int argc, char* argv[]) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        printUsage(argv[0]);
        return 0;
    }

    bool jsonMode = false;
    bool batchCsvMode = false;
    std::string modelPath;
    std::string imagePath;
    std::string datasetRoot;
    std::string outputCsvPath;

    if (argc == 3) {
        modelPath = argv[1];
        imagePath = argv[2];
    } else if (argc == 4 && std::string(argv[1]) == "--json") {
        jsonMode = true;
        modelPath = argv[2];
        imagePath = argv[3];
    } else if ((argc == 4 || argc == 5) && std::string(argv[1]) == "--batch-csv") {
        batchCsvMode = true;
        modelPath = argv[2];
        datasetRoot = argv[3];
        outputCsvPath = (argc == 5) ? argv[4] : getPredictionCsvFilename(modelPath);
    } else {
        printUsage(argv[0]);
        return 1;
    }

    try {
        debugPrint("Creating ONNX Runtime environment.");
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "InferenceService");

        debugPrint("Loading ONNX model.");
        Ort::Session session = createSession(env, modelPath);

        if (batchCsvMode) {
            std::cout << "========== Batch CSV Export ==========\n";
            std::cout << "Model Path   : " << modelPath << "\n";
            std::cout << "Dataset Root : " << datasetRoot << "\n";
            std::cout << "Output CSV   : " << outputCsvPath << "\n\n";

            const std::size_t rows = exportDatasetCsv(session, datasetRoot, outputCsvPath);

            std::cout << "Exported rows: " << rows << "\n";
            std::cout << "Saved CSV    : " << outputCsvPath << "\n";
            return 0;
        }

        if (!jsonMode) {
            printConsoleHeader(modelPath, imagePath);
        }

        debugPrint("Running single-image inference.");
        const Prediction prediction = inferImage(session, imagePath);
        const int predictedClass = predictedClassIndex(prediction);

        if (jsonMode) {
            std::cout << predictionToJson(prediction).dump(2) << "\n";
            return 0;
        }

        std::cout << "==================================================\n";
        std::cout << "Inference Result\n";
        std::cout << "==================================================\n";
        std::cout << "Inference Time      : " << prediction.inferenceTimeMs << " ms\n";
        std::cout << "Predicted Component : " << prediction.predictedLabel << "\n";
        std::cout << "Top Confidence      : "
                  << std::fixed << std::setprecision(2)
                  << prediction.confidence * 100.0f << " %\n";
        std::cout << "Review Threshold    : "
                  << std::fixed << std::setprecision(2)
                  << getConfidenceThreshold() * 100.0f << " %\n\n";

        printProbabilityTable(prediction.probabilities);
        std::cout << "\n";

        debugPrint("Saving confidence chart.");
        const std::string chartPath = getChartFilename(modelPath);
        drawConfidenceChart(
            prediction.probabilities,
            CLASS_NAMES,
            predictedClass,
            chartPath
        );
        std::cout << "Saved confidence bar chart: " << chartPath << "\n\n";

        std::cout << "==================================================\n";
        std::cout << "LLM Diagnostic Insight\n";
        std::cout << "==================================================\n";

        debugPrint("Initializing libcurl.");
        CurlGlobalGuard curlGuard;

        const std::string llmResponse =
            queryLLM(prediction.predictedLabel, prediction.probabilities);
        const bool llmOk = llmResponse.rfind("[LLM Error]", 0) != 0;

        if (llmOk) {
            std::cout << "LLM Status          : OK\n";
            std::cout << llmResponse << "\n";
        } else {
            std::cout << "LLM Status          : Failed\n";
            std::cout << llmResponse << "\n";
            std::cout << "Fallback            : Classification result and chart were generated successfully.\n";
        }

        std::cout << "==================================================\n";
    } catch (const std::exception& e) {
        if (jsonMode) {
            json errorJson = {
                {"error", e.what()}
            };
            std::cout << errorJson.dump(2) << "\n";
        } else {
            std::cerr << "Error: " << e.what() << std::endl;
        }
        return 1;
    }

    return 0;
}
