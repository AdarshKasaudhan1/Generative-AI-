# Identifying Bias in a Pretrained Generative AI Model

## Overview

This project investigates potential representational bias in a pretrained Generative AI model using GPT-2. It examines how a language model generates text in response to neutral prompts describing different professions.

The experiment focuses on identifying patterns in gender representation, professional appearance, language, and tone across generated responses.

## Objective

To identify and analyze observable patterns and potential stereotypes in GPT-2-generated text using profession-based prompts.

## Professions Analyzed

The experiment evaluates six professions:

- Doctor
- Nurse
- Business Executive
- Scientist
- Police Officer
- Criminal

Five responses are generated for each profession to observe recurring patterns.

## Technologies Used

- Python
- Jupyter Notebook
- GPT-2
- Hugging Face Transformers
- Pandas

## Methodology

1. Load the pretrained GPT-2 language model and tokenizer.
2. Define neutral, profession-based text prompts.
3. Generate five text responses for each profession.
4. Examine the generated text for gender representation and other observable characteristics.
5. Organize the observations into a structured table.
6. Interpret the patterns and discuss possible sources of bias.

## Key Observations

The experiment reported the following patterns:

- **Doctor:** Mostly male representation.
- **Nurse:** Mostly female representation, with caring and compassionate language.
- **Business Executive:** Mostly male representation, with authoritative language.
- **Scientist:** Mixed gender representation, with research-focused descriptions.
- **Police Officer:** Mostly male representation, with disciplined and serious descriptions.
- **Criminal:** Mostly male representation, with suspicious and negative descriptions.

These observations describe the outputs examined in this experiment and should not be interpreted as universal characteristics of GPT-2.

## Why Does AI Bias Matter?

Generative AI models learn statistical patterns from their training data. If that data reflects historical or societal stereotypes, the model may reproduce similar patterns in generated text.

Identifying these patterns is an important step toward developing more inclusive and responsible AI systems.

## Possible Improvements

- Use more diverse and representative training datasets.
- Evaluate outputs using fairness benchmarks.
- Compare results across multiple pretrained language models.
- Repeat experiments with different prompts and random seeds.
- Use structured evaluation criteria to reduce subjective interpretation.
- Audit generated outputs regularly.

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/generative-ai-bias-analysis.git
cd generative-ai-bias-analysis
```

### 2. Install dependencies

```bash
pip install jupyter transformers torch pandas
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the project notebook (`.ipynb`) and run its cells in order.

**Note:** The notebook should contain the model-loading, text-generation, and observation-analysis code. The exact notebook filename may differ.

## Limitations

- The experiment covers only six profession-based prompts.
- The observations are based on a limited number of generated responses.
- Outputs may change with sampling settings and random seeds.
- Gender and tone classifications may involve subjective judgments.
- The experiment identifies possible patterns but does not establish their statistical significance or prove the cause of bias.

## Reference

Bender, E. M., Gebru, T., McMillan-Major, A., & Shmitchell, S. (2021). *On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?* Proceedings of the ACM Conference on Fairness, Accountability, and Transparency (FAccT 2021).

## Conclusion

This project demonstrates how a pretrained generative language model can be examined for potential representational bias. Systematic evaluation, diverse training data, and ongoing fairness assessments can help support more balanced and responsible AI development.
