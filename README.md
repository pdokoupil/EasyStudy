# About
This is a repository with sources for the paper *EasyStudy: Framework for Easy Deployment of User Studies on Recommender Systems*

# About EasyStudy
A framework for quick & easy deployment of customizable RS user studies. The framework is modular, highly extensible and easy to use. It offers out-of-the-box functionality for all phases of the user study, i.e., data loading and presentation, preference elicitation, baseline recommending algorithms as well as result comparison layouts. It also features a  *fastcompare* plugin implementing the most common study flow, i.e., comparing perceived utility of several personalized recommenders.

# Contact
Patrik Dokoupil patrik.dokoupil@matfyz.cuni.cz
Ladislav Peska ladislav.peska@matfyz.cuni.cz

# Links
**If you want to try the framework on your own, please use the following links**.
- [Administration](https://tinyurl.com/EasyStudyAdmin)
- [Database](https://tinyurl.com/EasyStudyDb)

You can find multiple existing studies there and you can also create new studies as well.<br>

***Access details are in the paper**

Alternatively we have prepared a recording that shows a quick walkthrough of the framework, where we first show the "researcher view" to use Administration in order to create a simple user study, then show participant view (pass through the study itself) and briefly explain where the interactions can be found.
- [Quick walkthrough recording](https://tinyurl.com/EasyStudyDemo)


# Setup
If you want to quickly try the framework, feel free to use the links above. If you need to have a local deployment of the framework, details are given in this section:

We provide a [`Dockerfile`](./server/Dockerfile) for building an image with all the dependencies. To build the docker image, you can use the [`build_server_container.sh`](./server/build_server_container.sh) script.
To run the container, you may use the [`start_server.sh`](./server/start_server.sh) script. The default port is set to 5555, but you can change it by modifying the [`start_server.sh`](./server/start_server.sh) script and the [`Dockerfile`](./server/Dockerfile).

To start user studies (from *fastcompare*) successfully, you have to provide the datasets that are provided by *fastcompare* and any other datasets that you plan to support in your newly designed plugins/*fastcompare* data loader extensions. There is a small limitation, that images or files in Flask has to be in the *static* directory. It can be *static* directory inside one of the plugins or of the whole Flask app, and we decided to use the latter, so all images have to be put into [static/datasets/X/img](./server/static/datasets/X) where `X` denotes dataset name ({'ml-latest', 'goodbooks-10k'}). Note that although this may seem a bit restricting, in the end, it is not, because you can always put just symlinks there, that will point to arbitrary directory in your system. The exact images that were used during Demonstration are available in [static/datasets/ml-latest/ml_latest_img.zip](./server/static/datasets/ml-latest/ml_latest_img.zip) and [static/datasets/goodbooks-10k/goodbooks_img.zip](./server/static/datasets/goodbooks-10k/goodbooks_img.zip) and should be extracted into [static/datasets/ml-latest/img](./server/static/datasets/ml-latest/img) and [static/datasets/goodbooks-10k/img](./server/static/datasets/goodbooks-10k/img) respectively.

Datasets itself (csv files) can be downloaded from official links:
[ml-latest](https://files.grouplens.org/datasets/movielens/ml-latest.zip),
[goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k/archive/refs/heads/master.zip)
and the csv files should be extracted to [static/datasets/ml-latest/*.csv](./server/static/datasets/ml-latest/) and [static/datasets/goodbooks-10k/*.csv](./server/static/datasets/goodbooks-10k/) respectively.

Alternatively you can run the server without the docker container, by just relying on Flask and using `flask --debug run` or `flask run` from the [server](./server/) directory. However, in that case, you have to ensure to have all the dependencies installed on your system (you can try to mimick what the [`Dockerfile`](./server/Dockerfile) does with dependencies). Running the serveer this way is especially useful during development of new functionality.



# Development
## Extending fastcompare
The *fastcompare* plugin itself can be extended with new data loaders, recommendation algorithms or preference elicitation methods. The extension is done by adding new implementation as a class that is subclassing appropriate base classe. **The new class can be put either directly into `plugins/fastcompare/algo/*.py` or into separate plugins, e.g. `plugins/newplugin/*.py`**
### Adding new datasets
Subclass the `DataLoaderBase` and implement all the abstract methods and properties, those are described in the code comments (you can get inspired by e.g. `GoodbooksDataLoader`)

Then put your dataset folder under [static/datasets](./server/static/datasets/)
Later you can return the following `url_for('static', filename=f'datasets/goodbooks-10k/img/{item_id}.jpg')`
alternatively, you can always return remote urls ('http://...') but they are slowing down the system.

If you cannot add a new folder to root static (e.g. your deployment only gives permission to modify /plugins), you can include it directly within your new plugin, and then address the image files using:
`pm.emit_assets('pluginName', filename)` with `from app import pm` (again, make sure to check `has_app_context `even before importing `pm`!)

### Adding new algorithms
Subclass the `AlgorithmBase` base class and implement it. All abstract methods and properties are described in the code comments.


### Adding new preference elicitation methods
Subclass the `PreferenceElicitationBase` and implement it. All abstract methods and properties are described in the code comments.

## Extending EasyStudy (adding plugins)
Adding new plugin is done by adding new folder under server/plugins directory. This directory has several requirements (see Flask-PluginKit's documentation) and the plugin itself should expose the following endpoints in order to become a valid plugin:
- `/create` (typically renders some HTML page where user enters parameters) and in the end, it MUST invoke `/create-user-study` endpoint
- `/initialize` this will be invoked after `/create-user-study` has been called and this is the place where the plugin should perform its initialization. Any long running initialization should be done in the background daemon process, without blocking the request. Once the initialization is done, the plugin is responsible for marking the particular user study as `initialized=True` and `active=True`
- `/join` will be called when user attempts to join the user study, by following the generated URL link. This is where full control is passed to the plugin and plugin should decide future steps.

We have prepared a [minimal working plugin template](./server/plugins/empty_template) that you can use as a starting point when developing new plugins.

## Git clone
In the case that git clone fails with the message similar to the following:

*Error downloading object: server/static/datasets/goodbooks-10k/goodbooks_img.zip (1e41bd5): Smudge error: Error downloading server/static/datasets/goodbooks-10k/goodbooks_img.zip (1e41bd5bd92d60a0ca122c987ba6535d14ecc535a369dbb89c7858fdcfb64124): batch response: This repository is over its data quota. Account responsible for LFS bandwidth should purchase more data packs to restore access.*

you can use the following command to suppress the error.

On Mac/Linux use:
```
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/pdokoupil/EasyStudy.git
```
On Windows use:
```
cmd /C "$env:GIT_LFS_SKIP_SMUDGE='1' & git clone https://github.com/pdokoupil/EasyStudy.git"
```

In either case, you will not receive dataset images that are tracked by git-lfs and therefore you have to download them manually from the following links:

- [ml_latest_img.zip](http://herkules.ms.mff.cuni.cz/ligan/easystudy/ml_latest_img.zip)
- [goodbooks_img.zip](http://herkules.ms.mff.cuni.cz/ligan/easystudy/goodbooks_img.zip)

These .zip archives should then be extracted at right place: expected structure is ```server/static/datasets/ml-latest/img/*.jpg``` for the Movielens dataset (and similarly for the goodbooks).