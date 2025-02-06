# Potential and Limitations of LLMs for Augmenting Lexical Knowledge Bases

## Introduction

This project aims to explore the potential of Large Language Models (LLMs) in enhancing and expanding lexical knowledge bases (KBs). The project addresses the common limitations of KBs, such as static nature, limited coverage, and labor-intensive creation and maintenance. By leveraging LLMs, the project proposes a methodology to accurately reconstruct information from a source KB and generate new knowledge. The effectiveness of this methodology is evaluated using various LLMs and prompting techniques across three separate KBs.

## Folders  

This project is organized into the following folders:  

### src 

The src folder contains the runnable code wrote to submit the created prompts to a LLM instance and save the resulting outputs

### evaluation 

The evaluation folder contains: 
- **the results of the automatic evaluation**, thus the comparison between the semantic knowledge produced by the LLMs and that already present in the KBs
- **the results of the manual 'human-in-the-loop'evaluation**,  performed by three annotators on the ConceptNet and FrameNet's structured prompt results. These latter results are presented singularly for each annotator and are merged into two files: final_merge.jsonl contains the most-voted answer for each question evaluated by the annotators, aggregated_results.jsonl reorganizes the results of final_merge.jsonl aggregating them

### datasets 

The dataset folder contains:
- **the prompts created for four different KBs** (Semagram, MultiAligNet, ConceptNet, FrameNet)
- **the resulting answers given by each LLM** (c4ai-command-r-plus, gemma-2-27b-it, Jamba-v0.1, L3-8B-Stheno-v3.2, Meta-Llama-3-70B-AWQ, Meta-Llama-3-70B-Instruct-FP8, Mistral-7B-Instruct-v0.3, Phi-3-medium-4k-instruct)

### visualizations

The visualizations folder contains a graphic representation of the results obtained in the automatic evaluation

## License

This project is licensed under the GNU General Public License (GPL). 
The GPL license grants users the freedom to use, modify, and distribute the software under certain conditions. 
Please note that the GPL license is applicable to the project as a whole, including the code, documentation, and any associated materials.

You can find the full text of the license in the [LICENSE](LICENSE) file.
