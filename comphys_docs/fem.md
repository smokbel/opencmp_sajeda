## Finite Element References

*By Elizabeth Monte*



This document lists various references for the finite element method that I found useful when learning the method. The references are listed roughly in order of how I think they should be approached, but jump around or ask other people in the group for their recommendations if you don't find these references helpful.



In any cases below where I give a title and then a single author name with a year I am referring to the bibtexkey for an entry in opencmp.bib (repositories/references/ on Graham). If I give no author information the paper can be found in "Files".



****



**Applied Mathematics and Modelling for Chemical Engineers - Rice and Do**

*Chapter 8*

I recommend chapter 8 of this textbook as the starting point for learning the finite element method.



**fenics_book**

*Chapters 1-2*

The group has switched over from FEniCS to NGSolve, but I find the FEniCS documentation more understandable than the NGSolve documentation which is written for mathematicians. The FEniCS book is a good introduction to the practical use of finite element libraries. I recommend working through the first couple of examples (Poisson equation) to get a feel for how to code a simulation and then switching over to NGSolve for your actual work. FEniCS and NGSolve have different syntax, but the code structure will be very similar. WARNING: Make sure the version of FEniCS you install matches the version the documentation was written for; FEniCS changes dramatically between releases.



**The Finite Element Method: Theory, Implementation, and Applications - Larson and Bengzon**

*Entire Book*

I've only used this textbook as a reference for FEM, I haven't actually read the whole thing. That said, I would go through this reference after working through some FEniCS examples to get a more mathematically rigorous background on FEM.



**Finite Elements and Fast Iterative Solvers - Elman, Silvester, and Wathen**

*Chapters 1-2*

I personally would leave this textbook until last, but you could sub it in place of the Larson textbook. It is very math-heavy.