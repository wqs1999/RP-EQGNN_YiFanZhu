This project is organized as follows:

model/circuit: Contains individual modules of the model, such as the core structure for circuits or network components.
model/qgnn: Holds the primary implementation of the QGNN (Quantum Graph Neural Network).
train.py: The main training script. When run, it automatically downloads and preprocesses the QM9 dataset (if not already available) and then starts the training process.
Usage
Environment Setup

Create a virtual environment:
python -m venv venv
source venv/bin/activate  # On Linux/Mac
venv\Scripts\activate     # On Windows
Install the necessary dependencies using requirements.txt:
pip install -r requirements.txt
Data Download and Preprocessing

No manual download needed: The script train.py automatically triggers the download process via prepare/download and performs preprocessing via prepare/process_files.
Once the dataset is downloaded and processed, it will be stored in the appropriate folder specified in the code.
Run Training

In the command line, navigate to the project directory.
Execute:

python train.py
The script will handle dataset downloading (if necessary), preprocessing, and then proceed with training on the QM9 dataset.
Model Structure

model/circuit: Modular design for the internal structure of the model, making it easier to combine, maintain, and extend.
model/qgnn: Main QGNN implementation, integrating quantum modules with graph neural network functionality.
Results and Visualization

Check the console output for training logs, including metrics like loss and accuracy.
If you require visualization (e.g., using Matplotlib or TensorBoard), you can integrate these tools by updating the code accordingly.
