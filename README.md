## Prerequisites
1. Python 3.9.13
2. Make for Windows

## Installation
1. **Clone repository**
    ```
    git clone https://github.com/Anh26535D/automatic-mcqs-generation.git
    ```
2. **Train to get distractor generation model, you can find code in `/distractor_gen/kaggle/` folder, after training, put the LoRa checkpoint folder to `/distractor_gen/` folder**

3. **Create virtual environments for each module and install required libraries**
    ```
    python3 -m venv .venv
    ./.venv/Scripts/activate
    pip install -r requirements.txt

    cd distractor_gen
    python3 -m venv .venv
    ./.venv/Scripts/activate
    pip install -r requirements.txt
    
    cd..
    cd qa_gen
    python3 -m venv .venv
    ./.venv/Scripts/activate
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```

## Usage

1. **Open one terminal and enable the question generation module:**
    ```
    make run_qg
    ```

2. **Open a new terminal and run the distractor generation module:**
    ```
    make run_dg
    ```

3. **Open another terminal and run the main app:**
    ```
    make run_app
    ```

4. **Access the application:**
   Open your web browser and go to [http://localhost:5002](http://localhost:5002).