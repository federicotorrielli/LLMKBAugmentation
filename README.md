# LLM-Semagram  

LLM-Semagram is a research project aimed at extracting semantic knowledge from Large Language Models (LLMs) to scale existing knowledge bases. The project focuses on leveraging the power of LLMs to generate prompts for knowledge bases such as Semagram, MultiAligNet, and ConceptNet. 
The generated prompts are then used to query LLMs, which provide outputs that are evaluated using various metrics.  

## Folders  This project is organized into the following folders:  

### conceptnet extractor 

This folder contains the code used to create prompts from the ConceptNet knowledge base. 
The prompts generated from ConceptNet are an important input for the LLMs in this project.  

### evaluation 
The evaluation folder contains the code used to evaluate the models and the manual annotation. 
It includes scripts for calculating inter-annotation agreement and analyzing the results of the manual annotation process.  

### manual evaluation 

The manual evaluation folder contains the results and plots of the manual annotation. 
These results are crucial for understanding the performance of the LLMs and comparing them with the manual annotations.  

### prompts 

The prompts folder contains the prompts created for three knowledge bases: Semagram, MultiAligNet, and ConceptNet. 
These prompts are fed as input to the Large Language Models to generate meaningful outputs.  

### results 

The results folder contains three subfolders:  
- **prompt results**: This folder contains the output of the Large Language Models on the given prompts. It provides insights into how well the models can generate responses based on the prompts.
- **score results**: The score results folder contains the evaluation of the output using metrics such as Precision, Recall, Mean Reciprocal Rank, and Hits. These metrics help assess the quality and performance of the generated outputs.
- **annotation results**: This folder contains the results of the manual annotation process. It includes the annotated data and any analysis performed on it.  

## License  

This project is licensed under the GPL (GNU General Public License). 
You can find the full text of the license in the [LICENSE](LICENSE) file.
