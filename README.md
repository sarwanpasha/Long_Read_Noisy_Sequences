# Assessing the Resilience of Machine Learning Models on SARS-CoV-2 Genome Sequences Generated Using Long-read Specific Errors: A Study

The code for OHE-based embedding generation is given in ```OHE_Encoding.py``` python file

The code for WDGRL-based embedding generation is given in ```wdgrl_generation.py``` python file

The code for Spaced k-mers-based embedding generation is given in ```spaced_kmers_generation.py``` python file

The code for Weighted k-mers-based embedding generation is given in ```Weighted_kmers_IDF.py``` python file

The code for PWM-based embedding generation is given in ```PWM_kmers.py``` python file

To compute classification results, you can run ```classification.py``` python file

To compute robustness results, you can run ```robustness.py``` python file

The code for String Kernel is taken from the following repository
https://github.com/sarwanpasha/Approximate_Kernel

To run the code, you just need to update the dataset path in the python files and it will compute the embeddings.
