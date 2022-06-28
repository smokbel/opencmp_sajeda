## Discontinuous Galerkin References

*By Elizabeth Monte*



This document lists various references for the discontinuous Galerkin method that I found useful when learning the method. WARNING: You should absolutely not attempt to learn DG until you have a good understanding of FEM.



In any cases below where I give a title and then a single author name with a year I am referring to the bibtexkey for an entry in opencmp.bib (repositories/references/ on Graham). If I give no author information the paper can be found in "Files".



****



For a derivation of the DG formulations of the models used in OpenCMP see Appendix A of my thesis. It is probably the best starting point if you just want a condensed version of how our group uses DG.



**Discontinuous Galerkin Methods - cockburn_dg**

I believe this was paper written by Cockburn specifically to teach DG. I find the paper quite readable and it gives both an overview of the strengths and weaknesses of DG and a mathematical description of the method. I would start with this paper when learning DG. Just note that some terminology and notation differs from that used by the group.



**The Finite Element Method: Theory, Implementation, and Applications - Larson and Bengzon**

*Chapter 14*

This textbook gives a fairly technical overview of DG while being less math-heavy than a lot of papers on DG. I discovered it fairly late in my Master's, but I think it could give a good introduction to DG if gone through with someone who already understands DG. It does start out with a purely hyperbolic problem, which is not a use case our group usually considers. However, it is a simpler application of DG and possibly easier to understand than models that include diffusion.



**Discontinuous Galerkin Methods: Theory, Computation, and Applications - Cockburn, Karniadakis, and Shu**

*Part 1*

Part 1 gives a historical perspective on the use and development of DG. It's a fairly quick read and gives context for the use of DG.



**Unified Analysis of Discontinuous Galerkin Methods for Elliptic Problems - Arnold2002**

This paper was my starting point for learning DG. I would recommend starting with appendix A of my thesis since it removes some of the more confusing proofs and math-language then returning to this paper for a fully rigorous derivation.



**sander_navier_stokes**

This paper was a write-up by Sander to explain his formulation of interior penalty DG for the incompressible Navier-Stokes equations. Appendix A of my thesis and other derivations by our group more-or-less follow the same procedure. However, we do not use the same time discretization formulation as Sander. As far as I can tell, his scheme would not actually give second-order convergence and would not be possible to implement in NGSolve. James doesn't think Sander actually uses the scheme described in the paper, he thinks Sander just uses implicit Euler.
