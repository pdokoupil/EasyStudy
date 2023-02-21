
window.app = new Vue({
    el: '#app',
    delimiters: ['[[',']]'], // Used to replace double { escaping with double [ escaping (to prevent jinja vs vue inference)
    data: function() {
        const colsPerRow = itemsPerRow;

        var numAlgorithms = Object.keys(movies).length;
        var variantNames = new Array(numAlgorithms);
        let moviesColumnified = new Array(numAlgorithms);

        for (variantIdx in movies) {
            let variantResults = movies[variantIdx]["movies"];
            let order = parseInt(movies[variantIdx]["order"]);
            variantNames[order] = variantIdx.toUpperCase();
            
            let variantResultsColumnified = [];
            let row = [];
            for (idx in variantResults) {
                let movie = variantResults[idx];
                row.push(movie);
                if (row.length >= colsPerRow) {
                    variantResultsColumnified.push(row);
                    row = [];
                }
            }
            if (row.length > 0) {
                variantResultsColumnified.push(row);
            }
            moviesColumnified[order] = variantResultsColumnified;
        }

        let algorithmRatings = [];
        let algorithmRatingsValidated = [];
        for (let i = 0; i < numAlgorithms; ++i) {
            algorithmRatings.push(null);
            algorithmRatingsValidated.push(false);
        }
        return {
            variantsResults: moviesColumnified,
            selected: [],
            selectedMovieVariants: [],
            selectedMovieIndices: "",
            algorithmComparisonValue: numAlgorithms == 2 ? null : 0,
            algorithmComparisonValidated: numAlgorithms != 2, // For two algorithms, always mark it is nonvalid and wait for rating
            numAlgorithms: numAlgorithms,
            dontLikeAnythingValue: false,
            algorithmRatings: algorithmRatings,
            algorithmRatingsValidated: algorithmRatingsValidated,
            variantNames: variantNames,
            imageHeight: 300,
            maxColumnsMaxWidth: 300,
            busy: false,
            allAlgorithmRatingsValidated: false
        }
    },
    computed: {
        algorithmComparisonState() {
            this.algorithmComparisonValidated = this.algorithmComparisonValue != null;
            return this.algorithmComparisonValue != null;
        },
        dontLikeAnythingState() {
            return this.dontLikeAnythingValue;
        },
        allValidated() {
            let dontLikeAnythingValidated = this.selected.length > 0 || this.dontLikeAnythingValue;
            return this.algorithmComparisonValidated &&
                   this.allAlgorithmRatingsValidated && // All algorithms have their outputs rated
                   dontLikeAnythingValidated;
        }
    },
    methods: {

        // Custom (movie specific) implementation of indexOf operator
        // Considers only movie's properties
        movieIndexOf(arr, item) {
            for (let idx in arr) {
                let arrItem = arr[idx];
                if (arrItem.movie_idx === item.movie_idx
                    && arrItem.movie === item.movie
                    && arrItem.url === item.url) {
                        return idx;
                    }
            }
            return -1;
        },
        onSelectMovie(event, item, variant) {
            item['variant'] = variant;
            let index = this.movieIndexOf(this.selected, item);
            if (index > -1) {
                // Already there, remove it
                var copies = document.getElementsByName(event.srcElement.name);
                for (let j = 0; j < copies.length; ++j) {
                    copies[j].classList.remove("selected");
                }
                this.selected.splice(index, 1);
                reportDeselectedItem(`/utils/deselected-item`, csrfToken, item, this.selected, ()=>{ return {"variant": variant}; });
            } else {
                // Not there, insert
                var copies = document.getElementsByName(event.srcElement.name);
                for (let j = 0; j < copies.length; ++j) {
                    copies[j].classList.add("selected");
                }
                this.selected.push(item);
                reportSelectedItem(`/utils/selected-item`, csrfToken, item, this.selected, ()=>{ return {"variant": variant}; });
            }
            this.selectedMovieIndices = this.selected.map((x) => x.movie_idx).join(",");
            this.selectedMovieVariants = this.selected.map((x) => x.variant).join(",");
        },
        onAlgorithmRatingChanged(newRating, algorithmIndex) {
            let oldRating = this.algorithmRatings[algorithmIndex];
            this.algorithmRatingsValidated[algorithmIndex] = true;
            this.algorithmRatings[algorithmIndex] = newRating;
            this.allAlgorithmRatingsValidated = this.algorithmRatingsValidated.reduce((partialSum, a) => partialSum + a, 0) == this.numAlgorithms;
            reportOnInput("/utils/on-input", csrfToken, "rating", {
                "variant": algorithmIndex,
                "variant_name": this.variantNames[algorithmIndex],
                "old_rating": oldRating,
                "new_rating": newRating
            });
            this.$forceUpdate(); // Make sure it propagates to UI (some issues with modifying list via direct access)
        },
        algorithmRatingVariant(algorithmIndex) {
            if (this.algorithmRatingsValidated[algorithmIndex]) {
                return "success";
            }
            return "danger";
        },
        updateImageHeight() {
            if (window.innerHeight <= 750) {
                this.imageHeight = 150;
            } else if (window.innerHeight <= 950) {
                this.imageHeight = 200;
            } else {
                this.imageHeight = 300;
            }
        },
        updateMaxColumnsMaxWidth() {
            if (window.innerWidth <= 1300) {
                this.maxColumnsMaxWidth = 140;
            } else {
                this.maxColumnsMaxWidth = 300;
            }
        },
        updateResolutions() {
            this.updateImageHeight();
            this.updateMaxColumnsMaxWidth();
        }
    },
    async mounted() {
        const btns = document.querySelectorAll(".btn");
        const chckbxs = document.querySelectorAll("input[type=checkbox]");
        const radios = document.querySelectorAll("input[type=radio]");
        // Register the handlers for event reporting
        startViewportChangeReportingWithLimit(`/utils/changed-viewport`, csrfToken, 1.0, true, compare_ctx_lambda);
        startScrollReportingWithLimit(`/utils/changed-viewport`, csrfToken, 1.0, document.getElementsByName("scrollableDiv"), compare_ctx_lambda);
        registerClickedButtonReporting(`/utils/on-input`, csrfToken, btns);
        registerClickedCheckboxReporting("/utils/on-input", csrfToken, chckbxs);
        registerClickedRadioReporting("/utils/on-input", csrfToken, radios);
        reportLoadedPage(`/utils/loaded-page`, csrfToken, "compare_algorithms", ()=>
            {
                return {
                    "result_layout": resultLayout,
                    "movies": movies,
                    "iteration": iteration,
                    "min_iteration_to_cancel": minIterationToCancel
                };
            }
        );

        this.updateResolutions();
        window.addEventListener("resize", this.updateResolutions);
    }
})