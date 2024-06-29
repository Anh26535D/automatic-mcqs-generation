.PHONY: run_qg run_dg run_t5qg run_paraphrase

run_qg:
	cd qa_gen && .venv\Scripts\python.exe app.py

run_dg:
	cd distractor_gen && .venv\Scripts\python.exe app.py

run_t5qg:
	distractor_gen\.venv\Scripts\python.exe t5_qa_gen/app.py

run_paraphrase:
	distractor_gen\.venv\Scripts\python.exe paraphraser/paraphrase_app.py