# The-impact-of-contact-data-resolution-on-disease-transmission-intervention-strategy-evaluations
FYP for NTU (Bachelor of Computer Science)

Case senario.ipynb allows the simulation of a chosen specific case. all intervention parameters need to be entered 

simulation.py will run the whole simulation and create the necessary graph for all sensitivity analyses. However, due to the large number of iterations, a few hours or even days are needed to run the code.

inside the simulation file, i separated baseline analyses and each section of the sensitivity analysis into different Python files such that only the required section would be run. you may adjust the number of cores in the Quarantine.py file for the multiprocessing according to the device using. 
