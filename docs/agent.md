# Running with an agent

SSAF-T can be run end to end by an LLM agent through the **ssaft skill**, which is distributed
separately from this package. The agent:

1. checks the Python environment, and if `ssaft` is missing creates a virtual environment and
   installs it as in [Installation](installation.md) (GPU: Python 3.11 + CUDA 13 only);
2. inspects the dataset with `spatialdata_io` to identify the platform and the available images,
   points and shapes;
3. asks every run decision **once, before anything runs**. For Xenium, MERSCOPE and CosMx the first
   question is 2D or 3D. Every question's first option is the platform default, linked to its
   parameter on these pages, and any value can be typed in instead;
4. runs the stages for the platform in order ([Pipeline](pipeline/index.md)), monitoring logs and
   on-disk outputs;
5. at the annotation stage, starts a separate CeLLMan caller session, which spawns the
   `annotate-expert` agent. The SSAF-T agent does not annotate itself; it relays the caller's
   questions to you and waits for the annotated h5ad;
6. maps, filters and (optionally) segments, and reports where the outputs are.

Parameters marked **Asked: yes** in the pipeline pages are the ones the agent asks about. Everything
else stays at the platform default unless you name it in your request.
