# Extracting Semantic Knowledge from Large Language Models

## Introduction

This project aims to explore the potential of Large Language Models (LLMs) in enhancing and expanding lexical knowledge bases (KBs). The project addresses the common limitations of KBs, such as static nature, limited coverage, and labor-intensive creation and maintenance. By leveraging LLMs, the project proposes a methodology to accurately reconstruct information from a source KB and generate new knowledge. The effectiveness of this methodology is evaluated using various LLMs and prompting techniques across three separate KBs.

## Folders  

This project is organized into the following folders:  

### conceptnet extractor 

This folder contains the code for creating prompts from the ConceptNet knowledge base. 
The prompts are designed to elicit specific semantic information from the LLMs and serve as input for extracting knowledge. 

### evaluation 

The evaluation folder contains the code for evaluating the performance of the models and the manual annotation. 
It includes scripts for measuring inter-annotation agreement and analyzing the annotation results. 
The evaluation metrics used include Precision, Recall, Mean Reciprocal Rank, and Hits.

### manual evaluation 

The manual evaluation folder contains the results and plots of the manual annotation process. 
The manual annotation is conducted to assess the quality and accuracy of the extracted semantic knowledge. 
The folder includes annotated data, analysis scripts, and visualizations of the annotation results. 

### prompts 

The prompts folder contains the prompts created for three different knowledge bases: Semagram, MultiAligNet, and ConceptNet. 
These prompts are fed as input to the Large Language Models to extract semantic knowledge. 
Each subfolder corresponds to a specific knowledge base and contains the necessary prompt files.

### results 

The results folder consists of three subfolders: 

- **prompt results**: This folder contains the output of the Large Language Models on the given prompts. It includes the generated responses and extracted semantic information.
- **score results**: The score results folder contains the evaluation of the output using metrics such as Precision, Recall, Mean Reciprocal Rank, and Hits. These metrics provide insights into the accuracy and performance of the extracted knowledge.
- **annotation results**: The annotation results folder includes the results of the manual annotation process. It contains annotated data, statistical analysis, and visualizations of the annotation results.

## License

This project is licensed under the GNU General Public License (GPL). 
The GPL license grants users the freedom to use, modify, and distribute the software under certain conditions. 
Please note that the GPL license is applicable to the project as a whole, including the code, documentation, and any associated materials.

You can find the full text of the license in the [LICENSE](LICENSE) file.
