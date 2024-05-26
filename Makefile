.PHONY: run_qg run_dg run_app

run_qg:
	cd qa_gen && .venv\Scripts\python.exe app.py

run_dg:
	cd distractor_gen && .venv\Scripts\python.exe app.py

run_app:
	.venv\Scripts\python.exe app.py