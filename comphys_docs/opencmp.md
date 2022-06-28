## Using and Developing OpenCMP

*By Elizabeth Monte*



This document records practical knowledge for those in the group who would like to use or develop OpenCMP. It focuses on things only relevant to our group; see the main OpenCMP documentation for tutorials and examples.



This document is up-to-date as of August 2021. It is more-or-less my knowledge dump as I finish my Master's and leave the group, so it is highly likely that things will change in the future and this document may or may not get updated.



****



To understand the general purpose of OpenCMP, its desired capabilities, and its general structure see my thesis (euler branch at repositories/multiphase/research/theses/21_EM_MASc_Thesis/ on Graham). All of the references I used in my thesis are in the opencmp.bib file (repositories/references/ on Graham).



Alex and I have done the majority of the code development for OpenCMP. James has also contributed to the code and been heavily involved in constructing the models used in OpenCMP. All three of us are good references for how to use OpenCMP. Chahat and Ittisak are also familiar with using OpenCMP.



OpenCMP uses two different version control systems. 

The mercurial repository on Graham is used for day-to-day commits and includes various branches for different features under development. I own the main repository (repositories/code/opencmp). Anyone can clone the main repository, but I am the only one who can push changes to it. If you have changes you would like pushed to the main repository tell me and I will pull from your repository.

The GitHub repository is the publicly available version of OpenCMP. Currently Nasser, Alex, James, and I have permission to push directly to the GitHub repository; everyone else needs to fork it and issue a pull request. In practice, I'm the only one who pushes to the GitHub repository and I push the latest changes to the default branch of the mercurial repository every couple of weeks. If you do push changes to the GitHub repository please update the CHANGELOG (docs/getting_started/) first. Also, in general, only the default branch of the mercurial repository should be pushed.



The OpenCMP room on element is where most group discussion of OpenCMP takes place. Ask to be added to it if you will be working with OpenCMP.



I own a Trello board where Alex, James, and I document bugs and features that would be nice to add to OpenCMP and assign them to various people. Ask to be added to the Trello board if you will be developing OpenCMP.



There is a Slack channel used for communication with the NGSolve group. Ask to be added to it if you will be working with OpenCMP. At the moment we generally meet once a month and the Slack channel is mainly used to schedule meetings and exchange code.



Our collaborations with the NGSolve group are currently focused on developing good preconditioners for our flow models. The NGSolve group gave us some code for a multigrid preconditioner for the Stokes equations with the diffuse interface method. However, we have not yet been able to fully understand the code and incorporate it into our work. The code can be found in Files/ ("stokes-multi-block.py" and "MGPreconditioner.py").



Most of the group uses Pycharm as their IDE; you can get the professional version for free if you are a student and it integrates nicely with mercurial. I highly recommend Meld for managing merges with mercurial. You can set it up to be your default merge tool and it's a lot easier than command line tools. The OpenCMP documentation is a mix of Markdown and reStructuredText. I recommend using Typora for writing Markdown documents. We also have various mathematical derivations typed up in LaTeX. Overleaf is a good tool if you've never used LaTeX; it has a lot of templates and allows you to collaborate with other people ala Google Docs. Mathpix is also handy if you hate typing out equations in LaTeX.