const OBJECTIVE_CHANGE_PERIOD_SECONDS = 1;

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

        relevance = [];
        diversity = [];
        novelty = [];
        relevanceValue = [];
        diversityValue = [];
        noveltyValue = [];
        relevanceDelta = [];
        diversityDelta = [];
        noveltyDelta = [];
        newWeights = [];
        newRelevance = [];
        newDiversity = [];
        newNovelty = [];
        middleRelevance = [];
        middleDiversity = [];
        middleNovelty = [];
        boughtRelevance = [];
        boughtDiversity = [];
        boughtNovelty = [];
        budget = [];
        let algoNames = Object.keys(movies);
        for (let i in algoNames) { // Iterate over refinement algorithms
            const algoName = variantNames[i];
            relevance.push(defaultRelevance[algoName]);
            diversity.push(defaultDiversity[algoName]);
            novelty.push(defaultNovelty[algoName]);
            relevanceValue.push(null);
            diversityValue.push(null);
            noveltyValue.push(null);
            relevanceDelta.push("0");
            diversityDelta.push("0");
            noveltyDelta.push("0");
            newWeights.push("");
            newRelevance.push(0);
            newDiversity.push(0);
            newNovelty.push(0);
            middleRelevance.push(0.5);
            middleDiversity.push(0.5);
            middleNovelty.push(0.5);
            boughtRelevance.push(Math.round(defaultRelevance[algoName] * 10));
            boughtDiversity.push(Math.round(defaultDiversity[algoName] * 10));
            boughtNovelty.push(Math.round(defaultNovelty[algoName] * 10));
            budget.push(5);
        }

        console.log(refinementAlgorithms);
        console.log(refinementAlgorithms.length);

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
            allAlgorithmRatingsValidated: false,

            // Fine tuning stuff
            relevance: relevance,
            diversity: diversity,
            novelty: novelty,
            relevanceValue: relevanceValue,
            diversityValue: diversityValue,
            noveltyValue: noveltyValue,
            relevanceDelta: relevanceDelta,
            diversityDelta: diversityDelta,
            noveltyDelta: noveltyDelta,
            deltaDescription: [
                "No attention",
                "Much less attention",
                "Less attention",
                "Slightly less attention",
                "Keep same",
                "Slightly more attention",
                "More attention",
                "Much more attention",
                "Full attention"
            ],
            newWeights: newWeights,
            newRelevance: newRelevance,
            newDiversity: newDiversity,
            newNovelty: newNovelty,
            middleRelevance: middleRelevance,
            middleDiversity: middleDiversity,
            middleNovelty: middleNovelty,
            refinementAlgorithms: refinementAlgorithms,
            showRefinement: false,
            budget: budget,
            boughtRelevance: boughtRelevance,
            boughtDiversity: boughtDiversity,
            boughtNovelty: boughtNovelty,
            lastRelevanceChange: new Date(),
            lastDiversityChange: new Date(),
            lastNoveltyChange: new Date()
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

        onTrailerWatch(movieData) {
            reportOnInput("/utils/on-input", csrfToken, "link", movieData);
        },

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
        },

        // Fine tuning stuff
        clipAndRound(x) {
            let result = Number(x.toFixed(2));
            if (result < 0) {
                return 0.0;
            } else if (result > 1) {
                return 1.0;
            }
            return result;
        },
        onRelevanceChange(newRel, event) {
            let idx = event.target.dataset.idx;

            var newRelevance = parseFloat(newRel);
            var othersAccum = this.diversity[idx] + this.novelty[idx];

            if (othersAccum == 0) {
                return this.clipAndRound(newRelevance);
            }

            var diversityShare = this.diversity[idx] / othersAccum;
            var noveltyShare = this.novelty[idx] / othersAccum;

            if (newRelevance > this.relevance[idx]) {
                // Handle increase
                let diff = newRelevance - this.relevance[idx];
                
                this.diversity[idx] -= diversityShare * diff;
                this.novelty[idx] -= noveltyShare * diff;

                let totalAccum = newRelevance + this.diversity[idx] + this.novelty[idx];
                this.diversity[idx] = this.clipAndRound(this.diversity[idx] / totalAccum);
                this.novelty[idx] = this.clipAndRound(this.novelty[idx] / totalAccum);

                return this.clipAndRound(newRelevance / totalAccum); //this.relevance + diff;
            } else if (newRelevance < this.relevance[idx]) {
                // Handle decrease
                let diff = this.relevance[idx] - newRelevance;
                
                this.diversity[idx] += diversityShare * diff;
                this.novelty[idx] += noveltyShare * diff;

                let totalAccum = newRelevance + this.diversity[idx] + this.novelty[idx];
                this.diversity[idx] = this.clipAndRound(this.diversity[idx] / totalAccum);
                this.novelty[idx] = this.clipAndRound(this.novelty[idx] / totalAccum);


                return this.clipAndRound(newRelevance / totalAccum); //this.relevance + diff;
            }

            return this.clipAndRound(newRelevance);
        },
        popoverShown(event) {
            reportOnInput("/utils/on-input", csrfToken, "popover-shown", {
                "target": {
                    "id": event.target.id,
                    "name": event.target.name
                }
            });
        },
        popoverHidden(event) {
            reportOnInput("/utils/on-input", csrfToken, "popover-hidden", {
                "target": {
                    "id": event.target.id,
                    "name": event.target.name
                }
            });
        },
        itemMouseEnter(event) {
            reportOnInput("/utils/on-input", csrfToken, "mouse-enter", {
                "target": {
                    "id": event.target.id,
                    "name": event.target.name
                }
            });
        },
        itemMouseLeave(event) {
            reportOnInput("/utils/on-input", csrfToken, "mouse-leave", {
                "target": {
                    "id": event.target.id,
                    "name": event.target.name
                }
            });
        },
        logRelevanceChange(oldVal, newVal, idx) {
            let now = new Date();
            if ((now - this.lastRelevanceChange) / 1000 >= OBJECTIVE_CHANGE_PERIOD_SECONDS) {
                reportOnInput("/utils/on-input", csrfToken, "range", {
                    "old_value": oldVal,
                    "new_value": newVal,
                    "metric": "relevance",
                    "algo_idx": idx,
                    "variant_name": this.variantNames[idx]
                });
                this.lastRelevanceChange = now;
            }
        },
        logDiversityChange(oldVal, newVal, idx) {
            let now = new Date();
            if ((now - this.lastDiversityChange) / 1000 >= OBJECTIVE_CHANGE_PERIOD_SECONDS) {
                reportOnInput("/utils/on-input", csrfToken, "range", {
                    "old_value": oldVal,
                    "new_value": newVal,
                    "metric": "diversity",
                    "algo_idx": idx,
                    "variant_name": this.variantNames[idx]
                });
                this.lastDiversityChange = now;
            }
        },
        logNoveltyChange(oldVal, newVal, idx) {
            let now = new Date();
            if ((now - this.lastNoveltyChange) / 1000 >= OBJECTIVE_CHANGE_PERIOD_SECONDS) {
                reportOnInput("/utils/on-input", csrfToken, "range", {
                    "old_value": oldVal,
                    "new_value": newVal,
                    "metric": "novelty",
                    "algo_idx": idx,
                    "variant_name": this.variantNames[idx]
                });
                this.lastNoveltyChange = now;
            }
        },
        onRelevanceChange2(newRel, event) { // This version works that if one objective is increasing, other two are decreasing proportionally
            let idx = event.target.dataset.idx;
            let newRelevance = parseFloat(newRel);
            let othersAccum = this.diversity[idx] + this.novelty[idx];
            let oldRelevance = this.relevance[idx];

            if (othersAccum == 0) {
                //return this.clipAndRound(newRelevance);
                this.novelty[idx] = 0.001;
                this.diversity[idx] = 0.001;
                othersAccum = this.novelty[idx] + this.diversity[idx];
            }

            let diversityShare = this.diversity[idx] / othersAccum;
            let noveltyShare = this.novelty[idx] / othersAccum;
            
            if (newRelevance > this.relevance[idx]) {
                let diff = newRelevance - this.relevance[idx];
                let newDiv = this.diversity[idx] - diversityShare * diff;
                let newNov = this.novelty[idx] - noveltyShare * diff;

                let totalAccum = newRelevance + newDiv + newNov;

                this.diversity[idx] = parseFloat(newDiv / totalAccum);
                this.novelty[idx] = parseFloat(newNov / totalAccum);
                this.relevance[idx] = parseFloat(newRelevance / totalAccum);
                let result = parseFloat(newRel / totalAccum);
                this.logRelevanceChange(oldRelevance, result, idx);
                return result;
            } else if (newRelevance < this.relevance[idx]) {
                let diff = this.relevance[idx] - newRelevance;
                let newDiv = this.diversity[idx] + diversityShare * diff;
                let newNov = this.novelty[idx] + noveltyShare * diff;

                let totalAccum = newRelevance + newDiv + newNov;

                this.diversity[idx] = parseFloat(newDiv / totalAccum);
                this.novelty[idx] = parseFloat(newNov / totalAccum);
                this.relevance[idx] = parseFloat(newRelevance / totalAccum);
                let result = parseFloat(newRel / totalAccum);
                this.logRelevanceChange(oldRelevance, result, idx);
                return result; 
            }
            let result = parseFloat(newRel);
            this.logRelevanceChange(oldRelevance, result, idx);
            return result;
        },
        onDiversityChange(newDiv, event) {
            let idx = event.target.dataset.idx;

            var newDiversity = parseFloat(newDiv);
            var othersAccum = this.relevance[idx] + this.novelty[idx];

            if (othersAccum == 0) {
                return this.clipAndRound(newDiversity);
            }

            var relevanceShare = this.relevance[idx] / othersAccum;
            var noveltyShare = this.novelty[idx] / othersAccum;

            if (newDiversity > this.diversity[idx]) {
                // Handle increase
                let diff = newDiversity - this.diversity[idx];
                
                this.relevance[idx] -= relevanceShare * diff;
                this.novelty[idx] -= noveltyShare * diff;

                let totalAccum = newDiversity + this.relevance[idx] + this.novelty[idx];
                this.relevance[idx] = this.clipAndRound(this.relevance[idx] / totalAccum);
                this.novelty[idx] = this.clipAndRound(this.novelty[idx] / totalAccum);

                return this.clipAndRound(newDiversity / totalAccum);
            } else if (newDiversity < this.diversity[idx]) {
                // Handle decrease
                let diff = this.diversity[idx] - newDiversity;
                
                this.relevance[idx] += relevanceShare * diff;
                this.novelty[idx] += noveltyShare * diff;

                let totalAccum = newDiversity + this.relevance[idx] + this.novelty[idx];
                this.relevance[idx] = this.clipAndRound(this.relevance[idx] / totalAccum);
                this.novelty[idx] = this.clipAndRound(this.novelty[idx] / totalAccum);

                return this.clipAndRound(newDiversity / totalAccum);
            }

            return this.clipAndRound(newDiversity);
        },
        onDiversityChange2(newDiv, event) {
            let idx = event.target.dataset.idx;

            let newDiversity = parseFloat(newDiv);
            let othersAccum = this.relevance[idx] + this.novelty[idx];
            let oldDiversity = this.diversity[idx];

            if (othersAccum == 0) {
                //return this.clipAndRound(newDiversity);
                this.relevance[idx] = 0.001;
                this.novelty[idx] = 0.001;
                othersAccum = this.relevance[idx] + this.novelty[idx];
            }

            let relevanceShare = this.relevance[idx] / othersAccum;
            let noveltyShare = this.novelty[idx] / othersAccum;
            
            if (newDiversity > this.diversity[idx]) {
                let diff = newDiversity - this.diversity[idx];
                let newRel = this.relevance[idx] - relevanceShare * diff;
                let newNov = this.novelty[idx] - noveltyShare * diff;

                let totalAccum = newDiversity + newRel + newNov;

                this.relevance[idx] = parseFloat(newRel / totalAccum);
                this.novelty[idx] = parseFloat(newNov / totalAccum);
                this.diversity[idx] = parseFloat(newDiversity / totalAccum);
                let result = parseFloat(newDiv / totalAccum);
                this.logDiversityChange(oldDiversity, result, idx);
                return result;
            } else if (newDiversity < this.diversity[idx]) {
                let diff = this.diversity[idx] - newDiversity;
                let newRel = this.relevance[idx] + relevanceShare * diff;
                let newNov = this.novelty[idx] + noveltyShare * diff;

                let totalAccum = newDiversity + newRel + newNov;

                this.relevance[idx] = parseFloat(newRel / totalAccum);
                this.novelty[idx] = parseFloat(newNov / totalAccum);
                this.diversity[idx] = parseFloat(newDiversity / totalAccum);
                let result = parseFloat(newDiv / totalAccum);
                this.logDiversityChange(oldDiversity, result, idx);
                return result;
            }
            let result = parseFloat(newDiv);
            this.logDiversityChange(oldDiversity, result, idx);
            return result;
        },
        onNoveltyChange(newNov, event) {
            let idx = event.target.dataset.idx;

            var newNovelty = parseFloat(newNov);
            var othersAccum = this.relevance[idx] + this.diversity[idx];

            if (othersAccum == 0) {
                return this.clipAndRound(newNovelty);
            }

            var relevanceShare = this.relevance[idx] / othersAccum;
            var diversityShare = this.diversity[idx] / othersAccum;

            if (newNovelty > this.novelty[idx]) {
                // Handle increase
                let diff = newNovelty - this.novelty[idx];
                
                this.relevance[idx] -= relevanceShare * diff;
                this.diversity[idx] -= diversityShare * diff;

                let totalAccum = newNovelty + this.relevance[idx] + this.diversity[idx];
                this.relevance[idx] = this.clipAndRound(this.relevance[idx] / totalAccum);
                this.diversity[idx] = this.clipAndRound(this.diversity[idx] / totalAccum);

                return this.clipAndRound(newNovelty / totalAccum);
            } else if (newNovelty < this.novelty[idx]) {
                // Handle decrease
                let diff = this.novelty[idx] - newNovelty;
                
                this.relevance[idx] += relevanceShare * diff;
                this.diversity[idx] += diversityShare * diff;

                let totalAccum = newNovelty + this.relevance[idx] + this.diversity[idx];
                this.relevance[idx] = this.clipAndRound(this.relevance[idx] / totalAccum);
                this.diversity[idx] = this.clipAndRound(this.diversity[idx] / totalAccum);

                return this.clipAndRound(newNovelty / totalAccum);
            }

            return this.clipAndRound(newNovelty);
        },
        onNoveltyChange2(newNov, event) {
            let idx = event.target.dataset.idx;

            let newNovelty = parseFloat(newNov);
            let othersAccum = this.relevance[idx] + this.diversity[idx];
            let oldNovelty = this.novelty[idx];

            if (othersAccum == 0) {
                this.relevance[idx] = 0.001;
                this.diversity[idx] = 0.001;
                othersAccum = this.relevance[idx] + this.diversity[idx];
                //return this.clipAndRound(newNovelty);
            }

            let relevanceShare = this.relevance[idx] / othersAccum;
            let diversityShare = this.diversity[idx] / othersAccum;
            
            if (newNovelty > this.novelty[idx]) {
                let diff = newNovelty - this.novelty[idx];
                let newRel = this.relevance[idx] - relevanceShare * diff;
                let newDiv = this.diversity[idx] - diversityShare * diff;

                let totalAccum = newNovelty + newRel + newDiv;

                this.relevance[idx] = parseFloat(newRel / totalAccum);
                this.novelty[idx] = parseFloat(newNovelty / totalAccum);
                this.diversity[idx] = parseFloat(newDiv / totalAccum);
                let result = parseFloat(newNov / totalAccum);
                this.logNoveltyChange(oldNovelty, result, idx);
                return result;
            } else if (newNovelty < this.novelty[idx]) {
                let diff = this.novelty[idx] - newNovelty;
                let newRel = this.relevance[idx] + relevanceShare * diff;
                let newDiv = this.diversity[idx] + diversityShare * diff;

                let totalAccum = newNovelty + newRel + newDiv;

                this.relevance[idx] = parseFloat(newRel / totalAccum);
                this.novelty[idx] = parseFloat(newNovelty / totalAccum);
                this.diversity[idx] = parseFloat(newDiv / totalAccum);
                let result = parseFloat(newNov / totalAccum);
                this.logNoveltyChange(oldNovelty, result, idx);
                return result;
            }
            let result = parseFloat(newNov);
            this.logNoveltyChange(oldNovelty, result, idx);
            return result;
        },
        onRelevanceDeltaChange(newVal) {
            reportOnInput("/utils/on-input", csrfToken, "range", {
                "old_value": this.relevanceDelta,
                "new_value": newVal,
                "labels": this.deltaDescription,
                "metric": "relevance"
            });
            return newVal;
        },
        onDiversityDeltaChange(newVal) {
            reportOnInput("/utils/on-input", csrfToken, "range", {
                "old_value": this.relevanceDelta,
                "new_value": newVal,
                "labels": this.deltaDescription,
                "metric": "diversity"
            });
            return newVal;
        },
        onNoveltyDeltaChange(newVal) {
            reportOnInput("/utils/on-input", csrfToken, "range", {
                "old_value": this.relevanceDelta,
                "new_value": newVal,
                "labels": this.deltaDescription,
                "metric": "novelty"
            });
            return newVal;
        },
        onSubmit(event) {
            this.busy = true;

            event.preventDefault();
            
            const MAX_SHARE = 0.8; // We can increase at most by this share of the remaining and decrease atmost by this share of current value
            
            let weights = [];
            for (let i in this.refinementAlgorithms) {
                if (this.refinementAlgorithms[i] == 0) {
                    weights.push('');
                } else {
                    // let relDelta = parseFloat(this.relevanceDelta[i]);
                    // let relOfMaxShare = 0.25 * relDelta; 

                    // let divDelta = parseFloat(this.diversityDelta[i]);
                    // let divOfMaxShare = 0.25 * divDelta;

                    // let novDelta = parseFloat(this.noveltyDelta[i]);
                    // let novOfMaxShare = 0.25 * novDelta;

                    // let baseRel = relDelta > 0 ? ((1.0 - defaultRelevance) * MAX_SHARE) : defaultRelevance * MAX_SHARE;
                    // let baseDiv = divDelta > 0 ? ((1.0 - defaultDiversity) * MAX_SHARE) : defaultDiversity * MAX_SHARE;
                    // let baseNov = novDelta > 0 ? ((1.0 - defaultNovelty) * MAX_SHARE) : defaultNovelty * MAX_SHARE;

                    // this.newRelevance[i] = defaultRelevance + relOfMaxShare * baseRel;
                    // this.newDiversity[i] = defaultDiversity + divOfMaxShare * baseDiv;
                    // this.newNovelty[i] = defaultNovelty + novOfMaxShare * baseNov;

                    // let sum = this.newRelevance[i] + this.newDiversity[i] + this.newNovelty[i];

                    // this.newWeights[i] = `${this.newRelevance[i]/sum},${this.newDiversity[i]/sum},${this.newNovelty[i]/sum}`;
                    // console.log("New weights are: " + this.newWeights[i]);
                    this.newWeights[i] = `${this.relevance[i]},${this.diversity[i]},${this.novelty[i]}`;
                }
            }
            this.newWeights = this.newWeights.join(";");
            //this.$forceUpdate();
            this.$nextTick(() => {
                event.target.submit(); 
            });
        },
        onClickContinue() {
            this.showRefinement = true;
        },
        decrease(idx, obj) {
            if (obj == 'relevance') {
                this.boughtRelevance[idx] = this.boughtRelevance[idx] - 1;
            } else if (obj == 'diversity') {
                this.boughtDiversity[idx] = this.boughtDiversity[idx] - 1;
            } else if (obj == 'novelty') {
                this.boughtNovelty[idx] = this.boughtNovelty[idx] - 1;
            }
            this.budget[idx] = this.budget[idx] + 1;
            this.$forceUpdate();
        },
        increase(idx, obj) {
            if (obj == 'relevance') {
                this.boughtRelevance[idx] = this.boughtRelevance[idx] + 1;
            } else if (obj == 'diversity') {
                this.boughtDiversity[idx] = this.boughtDiversity[idx] + 1;
            } else if (obj == 'novelty') {
                this.boughtNovelty[idx] = this.boughtNovelty[idx] + 1;
            }
            this.budget[idx] = this.budget[idx] - 1;
            this.$forceUpdate();
        },
        showAndScroll() {
            this.showRefinement = true;
            this.$nextTick(() => {
                window.setTimeout(function(){
                    window.scrollTo(0, document.body.scrollHeight);
                }, 150);
            });
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