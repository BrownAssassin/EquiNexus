# Kaggle notebooks

Use Kaggle notebooks to access competition inputs and GPU accelerators. The validation baseline notebook clones the public repository at main, records the resolved Git commit, and imports the shared implementation from src/.

Attach the competition dataset in Kaggle and enable internet for the repository clone and dependency check. The baseline itself is CPU-based and does not require a GPU; use a GPU accelerator for later model experiments. Record the Kaggle notebook version, source commit, accelerator, dependency versions, split digest, and output checksums in the experiment log. Do not commit downloaded data, model weights, credentials, or generated submission files.
