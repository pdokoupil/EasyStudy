# About
This is a repository with sources that were used for running the study described in the paper *Comparing User Interfaces for Customizing Multi-Objective Recommender Systems*.
The development was done in the ecosystem of the EasyStudy framework where we implemented a new plugin providing custom functionality tailored for our user study.

All code that is related to this paper can be found in [slidershuffling](./server/plugins/slidershuffling/) (mainly) and [multiobjective](./server/plugins/multiobjective/)

# Abstract

The goal of Multi-Objective Recommender Systems (MORSs) is to adapt to the needs and preferences of the users
from different beyond-accuracy perspectives. When a MORS operates at the local level, it tailors its results to
the needs of each individual user. Recent studies have highlighted that, however, the self-declared propensity
of the users towards the different objectives does not always match with the characteristics of the accepted
recommendations. Therefore, in this study, we delve into different ways for users to express their preference
toward multi-objective goals and observe whether they have some impact on declared propensities and overall
user satisfaction. In particular, we explore four different user interface (UI) designs and perform a user study
focused on the interactions with both the UI and the recommendations. Results show that multiple UIs lead
to similar results w.r.t. usage statistics, but users’ perceptions of these UIs often differ. These results highlight
the importance of examining MORSs from multiple perspectives to accommodate the users’ actual needs when
producing recommendations. Study data and detailed results are available from https://osf.io/pbd54/.

# Contact
- [Patrik Dokoupil](mailto:patrik.dokoupil@matfyz.cuni.cz)
- [Ladislav Peska](mailto:ladislav.peska@matfyz.cuni.cz)

# Running the study
Even though our contribution (in terms of source code) lies in mainly in the [slidershuffling](./server/plugins/slidershuffling/) plugin we still want to make it easier to run the study. Therefore instead of only providing the plugins, this repository provides the whole EasyStudy Fork, including the plugins.
To run the study it is enough to follow EasyStudy documentation on How to create a user study, **in short**: start the EasyStudy server, navigate to *administration*, create an account there, and log in. From the administration, find "slidershuffling" plugin in the list of available plugins and click on create. This will show a creating page where you can set study parameters, we used k=10 and "Random" as a refinement layout. Other options can be kept as is. After creating, you will be redirected back to administration and once the study is ready, a unique URL to access it will be generated.

**For more details**, please consult [EasyStudy documentation](https://github.com/pdokoupil/EasyStudy?tab=readme-ov-file#setup) 