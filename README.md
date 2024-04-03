# Sawmill

UUtilizing system logs to predict future system failures and perform counterfactual analysis.

### Current demo

You can find a current notebook-based demo at [demo.ipynb](demo.ipynb).

If you would like to launch the full web-based demo instead, please follow the instructions [here](webapp/README.md).

### GPT prompts

You can find the prompts we use for GPT in the respective files:
- [Variable tagging (line 215)](src/sawmill/tag_utils.py#215)
- [AskGPT evaluation baseline (line 226)](src/sawmill/causal_discoverer.py#226)
- [GPT as a causal discovery method in Table 10 (line 138)](src/sawmill/causal_discoverer.py#138)


### Documentation

To view the documentation, run `mkdocs serve` from the root of this repo and open the corresponding page. 

You might need to install the following packages:
`pip install mkdocs-material mkdocs-gen-files mkdocs-literate-nav markdown_include pymdown-extensions markdown mkdocs-pymdownx Pygments mkdocs-jupyter mkdocstrings-python mkdocstrings mdx_include`

### OpenAI integration

Yo use the LLM-powered capabilites of Sawmill, please add a `.env` file to the root of this repo and define `OPENAI_API_KEY` appropriately.


