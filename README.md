## Prerequisites
1. Python 3.9.13

## Usage
1. Create virtual environments for each module and install required libraries.
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

2. Run services
- Open one terminal and enable question generation module
    ```
    make run_qg
    ```
- Open a new terminal and run distractor generation module
    ```
    make run_dg
    ```
- Open a new terminal and run app
    ```
    ./.venv/Scripts/activate
    python app.py
    ```