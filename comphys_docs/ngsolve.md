## NGSolve Resources

*By Elizabeth Monte*



This document lists various resources for learning to use NGSolve.



****



NGSolve is a finite element solver library. Most of the group uses it to code their simulations and OpenCMP is built on top of it. The website is <https://ngsolve.org/>.



NGSolve comes bundled with Netgen, a meshing software and graphical interface. Netgen is very handy for visualising simulation results. It also has some nice automated meshing and mesh optimisation routines for meshing CAD and .stl files. You can use Netgen to construct meshes from scratch, but it's limited to combinations of simple primitives (bricks, spheres, cylinders...) so most of the group uses Gmsh to construct complex meshes from scratch. Gmsh is also preferable if you need to mark specific mesh surfaces for boundary conditions or if you want structured meshes. Netgen is best at generating unstructured meshes and can't generate 3D hexahedral meshes at all.



If you intend to use Gmsh meshes with NGSolve read through "Using Gmsh with NGSolve" in the main OpenCMP documentation (docs/helpers/).



NGSolve is supported on Linux, MacOS, and Windows. Most of the group runs it on Linux (Ubuntu 20.04-21.04). Alex also runs NGSolve on MacOS and Chahat and Ittisak run it on Windows. As far as I know no one in the group uses the Anaconda build, so I have no idea how buggy it is. Alex has had to build NGSolve from source a couple of times, so ask him for help if you need to do the same.



If you want to run NGSolve on Graham you will need to install it yourself. Tanya created a script for installing NGSolve with MPI support ("install_ngsolve_graham.sh" in Files/).



The best starting point to learn NGSolve is their tutorials at <https://docu.ngsolve.org/latest/>. I recommend working through all of *1. Getting Started*. Also go through *4. Geometric Modelling and Mesh Generation* if you want to use Netgen for meshing. Then pick and choose from the rest of the topics depending on your specific application. WARNING: NGSolve documentation is rather sparse. If you have no experience with finite element libraries I would start with the FEniCS documentation and then shift over to NGSolve once you have a good idea of how to structure simulation code.



Docstrings are the only way to get documentation for many NGSolve functions (call `help(function)` in your Python interpreter).



The NGSolve forum at <https://ngsolve.org/forum/index> is your best resource for everything not covered in the tutorials. I don't know how the good the built-in search functionality is, but just googling "ngsolve \<your issue\>" will usually bring up relevant forum posts. The NGSolve developers are also very active on the forum; I usually get a response to questions within a day.