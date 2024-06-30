## Prerequisites
1. Python 3.9.13
2. Make for Windows

## Installation
1. **Clone repository**
    ```
    git clone https://github.com/Anh26535D/automatic-mcqs-generation.git
    ```

2. **Create virtual environments for each module and install required libraries**
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

    cd..
    cd t5_qa_gen
    python3 -m venv .venv
    ./.venv/Scripts/activate
    pip install -r requirements.txt
    ```

## Usage
You can follow the `kaggle/` to train the distractor model and T5 question generation model 

1. **Open one terminal and enable the question generation module:**
    ```
    make run_qg
    ```

2. **Open a new terminal and run the t5 question generation module:**
    ```
    make run_t5qg
    ```

3. **Open a new terminal and run the distractor generation module:**
    ```
    make run_dg
    ```

4. **Open a new terminal and run the paraphraser module:**
    ```
    make run_paraphrase
    ```

5. **Generate QA**
    ```
    python main.py
    ```