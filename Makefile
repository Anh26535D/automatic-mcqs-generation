.PHONY: run_qg run_dg

run_qg:
	cd qa_gen && .venv\Scripts\python.exe app.py

run_dg:
	cd distractor_gen && .venv\Scripts\python.exe app.py