# About
This branch contains a checkpoint of EasyStudy framework with plugins that were used during the user studies where data for the research paper entitled **'SM-RS: Single- and Multi-Objective Recommendations with Contextual Impressions and Beyond-Accuracy Propensity Scores'** were collected. Note that the sources themselves are NOT the main contribution of this paper and were published separately along with an analysis of the user study results. The main contribution of the resource paper is the dataset, which is available in [OSF repository](https://osf.io/hkzje/). We maintain this repository solely for reproducibility purposes.

# Contact
Feel free to email [Patrik Dokoupil](patrik.dokoupil@matfyz.cuni.cz) if you have any questions regarding the code in this repository. 
See the paper for contact details of other authors (Ladislav Peska, and Ludovico Boratto) of the paper

# Running the study
To run the study it is enough to follow EasyStudy documentation on How to create a user study, **in short**: start the EasyStudy server, navigate to *administration*, create an account there, and log in. From the administration, find "slidershuffling" or "multiobjective" (dataset contains data from both) plugin in the list of available plugins and click on create. This will show a creating page where you can set study parameters, we used k=10 and "Random" as a refinement layout (only applicable for "slidershuffling" plugin). Other options can be kept as is. After creating, you will be redirected back to administration and once the study is ready, a unique URL to access it will be generated.

**For more details**, please consult [EasyStudy documentation](https://github.com/pdokoupil/EasyStudy?tab=readme-ov-file#setup)