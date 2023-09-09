# About
This is a repository with sources for the paper *EasyStudy: Framework for Easy Deployment of User Studies on Recommender Systems*. This readme also temporarily serves as a mixed user/developer documentation.

# About EasyStudy
A framework for quick & easy deployment of customizable RS user studies. The framework is modular, highly extensible and easy to use. It offers out-of-the-box functionality for all phases of the user study, i.e., data loading and presentation, preference elicitation, baseline recommending algorithms as well as result comparison layouts. It also features a  *fastcompare* plugin implementing the most common study flow, i.e., comparing perceived utility of several personalized recommenders.

# Contact
- Patrik Dokoupil (*patrik.dokoupil* at *matfyz.cuni.cz*)
- Ladislav Peska (*ladislav.peska* at *matfyz.cuni.cz*)

# Links
**If you want to try the framework on your own, please use the following links**.
- [Administration](https://bit.ly/EasyStudyAdmin)
- [Database](https://bit.ly/EasyStudyDb)

You can find multiple (example) existing studies there and you can also create new studies as well.
<br>

***Access details (username and password) are on the poster** - but feel free to create new account if you do not have them (you would not see the existing studies though) or reach out to us.

Alternatively we have prepared a recording that shows a quick walkthrough of the framework, where we first show the "researcher view" to use Administration in order to create a simple user study, then show participant view (pass through the study itself) and briefly explain where the interactions can be found.
- [Quick walkthrough recording](https://bit.ly/EasyStudyDemo)

# Plugins
**See the paper for detailed description of existing plugins and their functionality.**
- [fastcompare](./server/plugins/fastcompare/) allows comparison of 2-3 algorithms on implicit feedback dataset. Is further extensible by providing classes implementing predefined baseclass (for pref. elicitation, algorithms, data loaders). **Note**: out of the plugins that are provided out-of-the-box, this is the "main" plugin that is intended to cover the most common scenario, where researchers are interested in just comparing 2 (A/B test) or 3 algorithms in a within-user manner. **For more complex visulizations/study flows you can create new plugin, e.g. based on empty_template plugin**.
- [layoutshuffling](./server/plugins/layoutshuffling/) a plugin that we have used for our internal user study. Is kept here for illustrational purposes.
- [utils](./server/plugins/utils/) contains only shared functionality, does not serve as "plugin template".
- [vae](./server/plugins/vae/) plugin providing MultVAE and StandardVAE algorithm implementations for the *fastcompare* plugin. Similarly to *utils* this plugin only contains shared functionality and cannot be used to create new user study. Note that these algorithms are itentionaly separated from *fastcompare* plugin itself since they have extra dependencies that only have to be installed when this plugin is going to be used.
- [empty_template](./server/plugins/empty_template/) i.e. the minimal working plugin, see below.

First two plugins are **study plugins** meaning they can be used to create actual instance of an user study. Both *utils* and *vae* are **utility plugins** meaning they only provide some extra functionality (e.g. algorithms) that can be reused from other plugins. The *empty_plugin* is empty (smallest working example of a plugin) plugin that can be used as an inspiration for creating new plugins.

# Setup
If you want to quickly try the framework, feel free to use the links above. If you need to have a local deployment of the framework, details are given in this section:

We provide a [`Dockerfile`](./server/Dockerfile) for building an image with all the dependencies. To build the docker image, you can use the [`build_server_container.sh`](./server/build_server_container.sh) script.
To run the container, you may use the [`start_server.sh`](./server/start_server.sh) script. The default port is set to 5555, but you can change it by modifying the [`start_server.sh`](./server/start_server.sh) script and the [`Dockerfile`](./server/Dockerfile).

To start user studies (from *fastcompare*) successfully, you have to provide the datasets that are provided by *fastcompare* and any other datasets that you plan to support in your newly designed plugins/*fastcompare* data loader extensions. There is a small limitation, that images or files in Flask has to be in the *static* directory. It can be *static* directory inside one of the plugins or of the whole Flask app, and we decided to use the latter, so all images have to be put into [static/datasets/X/img](./server/static/datasets/X) where `X` denotes dataset name ({'ml-latest', 'goodbooks-10k'}). Note that although this may seem a bit restricting, in the end, it is not, because you can always put just symlinks there, that will point to arbitrary directory in your system. The exact images that were used during Demonstration are available in [static/datasets/ml-latest/ml_latest_img.zip](./server/static/datasets/ml-latest/ml_latest_img.zip) and [static/datasets/goodbooks-10k/goodbooks_img.zip](./server/static/datasets/goodbooks-10k/goodbooks_img.zip) and should be extracted into [static/datasets/ml-latest/img](./server/static/datasets/ml-latest/img) and [static/datasets/goodbooks-10k/img](./server/static/datasets/goodbooks-10k/img) respectively.

Datasets itself (csv files) can be downloaded from official links:
[ml-latest](https://files.grouplens.org/datasets/movielens/ml-latest.zip),
[goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k/archive/refs/heads/master.zip)
and the csv files should be extracted to [static/datasets/ml-latest/*.csv](./server/static/datasets/ml-latest/) and [static/datasets/goodbooks-10k/*.csv](./server/static/datasets/goodbooks-10k/) respectively.

Alternatively you can run the server without the docker container, by just relying on Flask and using `flask --debug run` or `flask run` from the [server](./server/) directory. However, in that case, you have to ensure to have all the dependencies installed on your system (you can try to mimick what the [`Dockerfile`](./server/Dockerfile) does with dependencies). Running the server this way is especially useful during development of new functionality.

The **expected use case** is that a small group of researchers run a single instance of the EasyStudy framework and use them to create and run the user studies. 


# Fastcompare plugin
Build-in plugin that allows creating simple user studies while minimazing the amount of code changes that are necessary to run such a study. Fastcompare only covers the basic, yet most common scenario of comparing either 2 or 3 algorithms against each other on implicit feedback data. All the researcher has to do is click on Create from Administration page, fill in the parameters, start the study, and then share study URL with participants. Usually, researcher will want to compare their newly proposed algorithm against other algorithms - the only code change that is necessary in this case is to provide implementation of this newly proposed algorithm, see Development section on "Extending fastcompare" that follows below.

## Configuring fastcompare (available parameters) ##
- *Data Loader*: sets the underlying dataset to be used in the study. Currently supported are:
    - [Goodbooks-10k dataset](https://github.com/zygmuntz/goodbooks-10k)
    - Filtered ML-25M dataset: ML-25M dataset where some users and movies were removed (see [1] for details about filtering)
- *Preference Elicitation*: preference elicitation method determines which items (and how) will be selected and presented to users during the initial step of the user study, where we perform the pref. elicitation (i.e. trying to learn user preferences by asking them to select item they have enjoyed in the past). Currently supported values:
    - Popularity sampling: sample displayed items randomly weighted by their popularity. Number of samples can be sat as another parameter.
    - Popularity sampling from buckets: items are sorted by popularity and divided into buckets. For each bucket we then sample items on random, weighted by their popularity. At the end, items sampled from all buckets are merged. Number of buckets and number of samples per bucket can be set as additional parameters.
    - Multi Objective Sampling from buckets: similar to above, but items are divided into buckets w.r.t. relevance, diversity, and novelty. See [1] for detailed description.
- *Recommendation size*: how many items each recommendation algorithm should recommend in each session/iteration (i.e. length of top-k). Number of buckets and number of samples per bucket can be set as additional parameters.
- *Prolific code*: completion code for prolific.co service, for the case participants will be hired from the service.
- *Number of algorithms*: number of algorithms that will be compared (2 or 3).
- *Number of iterations*: how many sessions/iterations of repeated recommendations will be shown to the participats.
- *Recommendation layout*: determines how the recommendation lists from individual algorithms will be displayed to the user. Available options: {rows, single-row, columns, single-column, single-scrollable-row, maximum-columns-per-width}, their detailed description can be found in [1].
- *Shuffle algorithms*: if true, algorithm placement on the page will be chosen randomly (e.g. for single-column and 2 algorithms, it will be determined randomly whether Algorithm A's recommendation list shows in left or right column)
- *Shuffle recommendations*: if true, top-k recommendation lists will be shuffled
- *Add final questionnaire*: allows to upload file with final questionnaire, see Additional data section below.
- *Override* "X": allows to override text in different text areas that are shown to the participants during the study. This is very useful for customization of the user study. One can e.g. change the informed consent that is shown to users at the beginning of the user study, or even extend it with images that will present detailed participant instructions.
- *Show final statistics*: whether final statistics should be shown to the user upon finishing the study.


# Development
## Extending fastcompare
The *fastcompare* plugin itself can be extended with new data loaders, recommendation algorithms, preference elicitation methods, or evaluation metrics. The extension is done by adding new implementation as a class that is subclassing appropriate base classe. **The new class can be put either directly into `plugins/fastcompare/algo/*.py` or into separate plugins, e.g. `plugins/newplugin/*.py`**
### Adding new datasets (Data Loaders)
Subclass the `DataLoaderBase` and implement all the abstract methods and properties, those are described in the code comments (you can get inspired by e.g. `GoodbooksDataLoader`)

Then put your dataset folder under [static/datasets](./server/static/datasets/)
Later you can return the following `url_for('static', filename=f'datasets/goodbooks-10k/img/{item_id}.jpg')`
alternatively, you can always return remote urls ('http://...') but they are slowing down the system.

If you cannot add a new folder to root static (e.g. your deployment only gives permission to modify /plugins), you can include it directly within your new plugin, and then address the image files using:
`pm.emit_assets('pluginName', filename)` with `from app import pm` (again, make sure to check `has_app_context `even before importing `pm`!)

**Note** since you have to provide class implementation when adding new dataset/data loader, the (persisted) data itself can have arbitrary format and it is up to the data loader implementation to parse it. This is a tradeoff between flexibility (being able to use arbitrary data format) and convenience (not having to touch the code). If you have a data that has exactly same format as data parsed by any of existing loaders, the provided implementation can simply call existing loader with new data.

### Adding new algorithms
Subclass the `AlgorithmBase` base class and implement it. All abstract methods and properties are described in the code comments (please refer to [algorithm_base.py](./server/plugins/fastcompare/algo/algorithm_base.py)) and omitted here for clarity.


### Adding new preference elicitation methods
Subclass the `PreferenceElicitationBase` and implement it. All abstract methods and properties are described in the code comments (please refer to [algorithm_base.py](./server/plugins/fastcompare/algo/algorithm_base.py)) and omitted here for clarity.

### Adding new evaluation metrics
Subclass the `EvaluationMetricBase` and implement it. All abstract methods and properties are described in the code comments (please refer to [algorithm_base.py](./server/plugins/fastcompare/algo/algorithm_base.py)) and omitted here for clarity.

## Extending EasyStudy (adding plugins)
Adding new plugin is done by adding new folder under server/plugins directory. This directory has several requirements (see Flask-PluginKit's documentation) and the plugin itself should expose the following endpoints in order to become a valid plugin:
- `/create` (typically renders some HTML page where user enters parameters) and in the end, it MUST invoke `/create-user-study` endpoint
- `/initialize` this will be invoked after `/create-user-study` has been called and this is the place where the plugin should perform its initialization. Any long running initialization should be done in the background daemon process, without blocking the request. Once the initialization is done, the plugin is responsible for marking the particular user study as `initialized=True` and `active=True`
- `/join` will be called when user attempts to join the user study, by following the generated URL link. This is where full control is passed to the plugin and plugin should decide future steps.
- `/results` (optional) is called upon clicking on "Results" in administration UI. If the implementation is not provided, fallback endpoint from the `utils` plugin is used. Being able to hook custom `/results` endpoint is very useful because plugins with specific flow may require very individual and customized evaluation strategies that cannot by captured easily by common interfaces (e.g. `EvaluationMetricBase`).

We have prepared a [minimal working plugin template](./server/plugins/empty_template) that you can use as a starting point when developing new plugins.

## Additional data ##
The [data](./data) directory contains an example of [(dummy) final questionnaire](./data/sample_questionnaire.html) that can be uploaded and used during study creation. This particular example was used in the demo video recording.


## References ##
[1] Patrik Dokoupil, Ladislav Peska, and Ludovico Boratto. 2023. Rows or Columns? Minimizing Presentation Bias When Comparing Multiple Recommender Systems. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '23). Association for Computing Machinery, New York, NY, USA, 2354–2358. https://doi.org/10.1145/3539618.3592056