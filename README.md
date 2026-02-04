# AI-Based Smart Waste Segregation System

An intelligent waste classification system that uses deep learning and computer vision to classify waste items in real-time using a webcam.

## 📜 Overview
This project aims to automate the process of waste segregation, which is a critical step in effective recycling and waste management. It utilizes a Convolutional Neural Network (CNN) to identify and classify different types of waste materials from a live video feed. The system is built with Python and TensorFlow, making it accessible and adaptable.

The current model is trained to classify items into six common categories:
- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash (Miscellaneous)

## 📊 Dataset Used
The model was trained on the **TrashNet** dataset, created by [garythung](https://github.com/garythung/trashnet).
- **Source:** The dataset is publicly available on [Hugging Face](https://huggingface.co/datasets/garythung/trashnet).
- **Content:** It consists of 2,527 images, resized to 512x384, distributed across the six categories mentioned above.
- **License:** The dataset is available under an MIT License.

## 🛠️ Technology Stack
- **Language:** Python 3.8+
- **Core Frameworks:**
  - [TensorFlow](https://www.tensorflow.org/): For building and training the deep learning model.
  - [OpenCV](https://opencv.org/): For accessing the webcam and real-time image processing.
- **Libraries:**
  - [NumPy](https://numpy.org/): For numerical operations.
  - [SciPy](https://scipy.org/): For various scientific computing tasks.

## 🤖 Model Architecture
The system uses **Transfer Learning** with the **MobileNetV2** architecture, pre-trained on the ImageNet dataset. The final classification layers of the base model were removed and replaced with a custom head:
1.  A `GlobalAveragePooling2D` layer to reduce spatial dimensions.
2.  A `Dense` layer with 1024 units and a 'relu' activation function.
3.  A final `Dense` (output) layer with 6 units (one for each class) and a 'softmax' activation for classification.

The base MobileNetV2 layers were frozen during training, and only the custom head was trained. This approach significantly speeds up training time while leveraging the powerful feature extraction capabilities of a state-of-the-art model.

## 🚀 Setup and Installation

Follow these instructions to set up the project on your local machine.

### Step 1: Clone the Repository
First, clone this repository to your computer.
```bash
git clone https://github.com/Yashraj-Jangra/waste-segregation-project.git
cd waste-segregation-project
```

### Step 2: Create and Activate a Virtual Environment
It is highly recommended to use a virtual environment to manage project dependencies.

**On Windows:**
```cmd
# Create the environment
python -m venv venv

# Activate the environment
.\venv\Scripts\activate
```

**On macOS / Linux:**
```bash
# Create the environment
python3 -m venv venv

# Activate the environment
source venv/bin/activate
```
After activation, you should see `(venv)` at the beginning of your terminal prompt.

### Step 3: Install Dependencies
With the virtual environment active, install all the required libraries from the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

## ⚙️ Usage
Once the setup is complete, you can run the live classification script. Make sure your webcam is connected and not being used by another application.

From the project's root directory (with the virtual environment still active), run the following command:
```bash
python live_classifier.py
```
A window should pop up showing your webcam feed. Point an object towards the camera, and the model will display the predicted class and confidence score in the top-left corner.

Press the **'q'** key on your keyboard to close the video window and stop the script.

## 📁 Project Structure
```
.
├── .gitignore               # Specifies files for Git to ignore.
├── live_classifier.py       # Script to run the live webcam classification.
├── prepare_model.py         # Script for data loading and model architecture definition (for reference).
├── train_model.py           # Script to train the model (for reference).
├── requirements.txt         # Lists all Python dependencies for the project.
├── waste-segregation-project.md # Initial detailed project plan.
└── waste_classifier_model.h5  # The final trained and saved model file.
```
