# 🐘 AI-Powered Elephant Detection for HEC Mitigation

This project is a complete, end-to-end prototype for identifying elephants in aerial imagery to aid in the proactive mitigation of Human-Elephant Conflict (HEC). It features a trained PyTorch object detection model, a full training and data-cleaning pipeline, and a user-friendly web application for real-time inference.


## 📖 Project Overview

Human-Elephant Conflict (HEC) is a critical issue in forest-fringe communities, like those in Tamil Nadu, where elephant herds enter agricultural lands, causing significant crop damage and creating dangerous situations. Forest Department teams, tasked with monitoring vast areas, are often limited by manpower and can only react *after* an incursion has occurred.

This project provides a technological solution for **proactive monitoring**. By using a deep learning model to scan aerial images (from sources like drones or satellites), it can automatically identify elephants, allowing patrol teams to anticipate high-risk zones, deploy resources more effectively, and intervene *before* conflicts escalate.

## ✨ Key Features

- **Trained AI Model:** A Faster R-CNN model (using a ResNet50 backbone) trained with PyTorch to specifically detect elephants.
- **Data Cleaning:** Includes a robust "deep scan" script to find and remove corrupted files from image datasets, ensuring training stability.
- **Complete Training Pipeline:** A Jupyter Notebook (`detection.ipynb`) with the full workflow for data loading, training, and loss analysis.
- **Interactive Web App:** A web application built with FastAPI and a modern HTML/JS frontend that allows users to upload an image and receive a prediction with visual bounding boxes.
- **Enhanced UI:** The frontend includes image zoom functionality and a clean, responsive design for a great user experience.

## 🛠️ Tech Stack

- **Machine Learning:** PyTorch
- **Object Detection Model:** Faster R-CNN (pre-trained on COCO)
- **Backend:** FastAPI, Uvicorn
- **Image Processing:** OpenCV, Pillow (PIL)
- **Data Handling:** Pandas, NumPy
- **Frontend:** HTML, CSS (Pico.css), JavaScript
- **Notebook:** Jupyter Notebook / VS Code

## 🚀 Setup and Installation

Follow these steps to set up the project environment.

### 1. Clone the Repository

(If you have this on Git, otherwise skip to step 2)
git clone [\[YOUR_REPOSITORY_URL\]](https://github.com/Harishankar00/Elephant_Detection)
cd Elephant_Detection



### 2. Create a Virtual Environment (Recommended)

python3 -m venv venv
source venv/bin/activate


### 3. Install Dependencies

This project requires several Python libraries. You can install them all using this command, or create a requirements.txt file.

pip install torch torchvision
pandas numpy
opencv-python-headless
matplotlib tqdm
fastapi "uvicorn[standard]"
python-multipart aiofiles pillow



### 4. Download the Dataset

This project uses the Aerial Elephant Dataset from Kaggle:  
https://www.kaggle.com/datasets/davidrpugh/aerial-elephant-dataset

- Download the dataset.
- Unzip the contents.
- Place all the files (the `.csv` files and the `training_images/test_images` folders) into the `dataset/` directory as shown in the file structure.

---

## ⚙️ Execution

There are two main parts to this project: (1) Training a new model and (2) Running the pre-trained web application.

### Part 1: (Optional) Re-training the Model

The `detection.ipynb` notebook contains the complete, step-by-step workflow for data cleaning, loading, training, and analysis.

**Open and Run the Notebook:**

jupyter notebook detection.ipynb



(Or open it in VS Code.)

- **Data Cleaning:** Run the "Deep Scan" cells to verify the integrity of your downloaded image files. The cleaning script will automatically remove any corrupted files found.
- **Run All Cells:** Run the cells in order to:
  - Load and prepare the data.
  - Define the ElephantDataset class.
  - Instantiate the model and optimizer.
  - Run the full 20-epoch training loop (this will take several hours on a GPU).
- **Analyze Results:** The notebook will save all model checkpoints (e.g., `model_epoch_1.pth`, etc.) in the `/outputs` folder and plot a loss curve graph. We analyzed this graph and determined `model_epoch_5.pth` to be the best model.

### Part 2: Running the Web Application (Main)

This runs the interactive web app using the best pre-trained model (`model_epoch_5.pth`).

- **Ensure the Model is in Place:** Make sure your best model checkpoint (e.g., `model_epoch_5.pth`) is located in the `/outputs` directory.
- **Start the Uvicorn Server:** In your terminal, from the `Elephant_Detection/` root directory, run:

uvicorn app:app --reload



- `app:app` tells Uvicorn to look for the file `app.py` and the FastAPI object `app`.
- `--reload` enables hot-reloading, so the server restarts automatically when you save changes to `app.py`.

- **Open the Application:** Once the server is running, open your web browser and go to:  
  http://127.0.0.1:8000

- **Use the App:** You will see the web UI. You can now upload an image to get a real-time elephant detection!

---

## 📈 Project Demo

Our model's training graph shows classic overfitting after Epoch 5, which is why we selected it as our best model.  
An example of the model's output, with green boxes representing the AI's predictions.

---

## 🌟 Acknowledgments

This project was made possible by the Aerial Elephant Dataset available on Kaggle.  
This prototype was built with the guidance of the Gemini AI assistant.