# Paddle PP-Human ONNX

Paddle PPHuman ONNX refers to the conversion of the PaddlePaddle's PPHuman human pose estimation model into the ONNX (Open Neural Network Exchange) format. ONNX is an open standard for representing machine learning models, allowing seamless interoperability between different deep learning frameworks. By converting Paddle PPHuman to ONNX, the model becomes accessible and deployable across a wider range of platforms and tools. This enables developers to utilize Paddle PPHuman's advanced human pose estimation capabilities in their projects regardless of the framework they are using, fostering innovation and efficiency in various computer vision applications.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Configuring Parameters](#configuring-parameters)
  - [Running test_video.py](#running-test_video.py)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Provide a brief overview of your project. Explain what the purpose of the project is and what it aims to achieve. Highlight any important features or concepts.

## Getting Started

In this section, you will find instructions on how to set up your environment and run the project.

### Prerequisites

List any prerequisites that users need to have installed before they can use your project. For example:

- Python 3.6 <=
- CUDA (if using GPU)
- CUDNN (if using GPU)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/qwertyz15/Paddle-PPHuman-Onnx.git
   ```

2. Change into the project directory:

   ```bash
   cd Paddle-PPHuman-Onnx
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```
## Downloading the Model

You can download the Paddle PPHuman ONNX model from [this drive link](https://drive.google.com/file/d/15AnT_YUtspmGN2A-wD1OTowN_T4muKq5/view?usp=sharing).

## Usage

In this section, explain how users can use your project. Provide specific instructions for configuring parameters using the `config.ini` file and running `test_video.py`.

### Configuring Parameters

Before running the `test_video.py` script, you can configure the following parameters in the `config.ini` file:

- `model`: Specify the ONNX model file to use. Default is `pphuman.onnx`.
- `video_url`: Specify the URL of the video to process. Default is `0` (use webcam).
- `cuda`: Set to `True` to use GPU acceleration (requires CUDA and CUDNN), or `False` to use CPU.

To modify these parameters:

1. Open the `config.ini` file in a text editor.

2. Update the values according to your requirements.

3. Save the file.

### Running test_video.py

To run the `test_video.py` script, follow these steps:

1. Make sure you have installed the required packages as mentioned in the [Installation](#installation) section.

2. Open a terminal and navigate to the project directory:

   ```bash
   cd /path/to/your-repo
   ```

3. Run the script:

   ```bash
   python test_video.py
   ```

## Contributing

If you would like to contribute to this project, feel free to open an issue or submit a pull request. We welcome contributions from the community!

## License

This project is licensed under the [MIT License](LICENSE).

---
# Paddle-PPHuman-Onnx
