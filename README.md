# Autograd Lagrangian Parameter Extraction
This started as a project for Portland State PHY322 Spring 2025, applying modern machine-learning techniques to simulations of the Lagrangian formulation of classical mechanical systems. Everything after the "May 6" report was done independently outside of the scope of the class

At the top level of the repository, you'll find two .pdf files ("May6_weekly" and "May13_weekly") -- these contain detailed summaries of the results of this project and are best read sequentially.

The final presentation for PHY322 is contained in /LaTeX/Class Presentation/ and contains a the same pair of reports, a PowerPoint presentation of background info, and several Jupyter notebooks with examples of the algorithm applied to a number of example physical systems:

    -Simple harmonic oscillator
    
    -2 particles in a harmonic potential with Coulomb interaction
    
        -Learning multiple parameterizations of the system: exact Lagrangian, polynomial coupling (sum of products of powers of positions), neural network coupling

Code located in "Code" folder. Everything here is written in Python, using PyTorch for the numerical portions.

Documents/drafts located in the "LaTeX" folder.

Undocumented progress will be added to future reports.
