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

To start user studies (from *fastcompare*) successfully, you have to provide the datasets that are provided by *fastcompare* and any other datasets that you plan to support in your newly designed plugins/*fastcompare* data loader extensions. There is a small limitation, that images or files in Flask has to be in the *static* directory. It can be *static* directory inside one of the plugins or of the whole Flask app, and we decided to use the latter, so all images have to be put into [static/datasets/X/img](./server/static/datasets/X) where `X` denotes dataset name ({'ml-latest', 'goodbooks-10k'}). Note that although this may seem a bit restricting, in the end, it is not, because you can always put just symlinks there, that will point to arbitrary directory in your system.

Alternatively you can run the server without the docker container, by just relying on Flask and using `flask --debug run` or `flask run` from the [server](./server/) directory. However, in that case, you have to ensure to have all the dependencies installed on your system (you can try to mimick what the [`Dockerfile`](./server/Dockerfile) does with dependencies). Running the serveer this way is especially useful during development of new functionality.