# About
This is a repository with sources that were used for running the study described in the paper *Comparing User Interfaces for Fine-Tuning Multi-Objective Recommender Systems*.
The development was done in the ecosystem of the EasyStudy framework where we implemented a new plugin providing custom functionality tailored for our user study.
We are sharing the link to this snapshot of a branch with the state from when the study was running for peer-review purposes.

All code that is related to this paper can be found in [slidershuffling](./server/plugins/slidershuffling/) (mainly) and [multiobjective](./server/plugins/multiobjective/)

# Contact
Anonymized for peer-review

# Running the study
Even though our contribution (in terms of source code) lies in those two plugins mentioned above ([slidershuffling](./server/plugins/slidershuffling/), [multiobjective](./server/plugins/multiobjective/))
we still want to make it easier to run the study. Therefore instead of only providing the plugins, this repository provides the whole EasyStudy Fork, including the plugins.
To run the study it is enough to follow EasyStudy documentation on How to create a user study, **in short**: start the EasyStudy server, navigate to *administration*, create an account there, and log in. From the administration, find "slidershuffling" plugin in the list of available plugins and click on create. This will show a creating page where you can set study parameters, we used k=10 and "Random" as a refinement layout. Other options can be kept as is. After creating, you will be redirected back to administration and once the study is ready, a unique URL to access it will be generated.

**For more details**, please consult [EasyStudy documentation](https://github.com/pdokoupil/EasyStudy?tab=readme-ov-file#setup) 