window.app = new Vue({
    el: '#app',
    delimiters: ['[[',']]'], // Used to replace double { escaping with double [ escaping (to prevent jinja vs vue inference)
    data: function() {
        
        return {
            relevance: defaultRelevance,
            diversity: defaultDiversity,
            novelty: defaultNovelty,
            relevanceValue: null,
            diversityValue: null,
            noveltyValue: null,
            relevanceDelta: "0",
            diversityDelta: "0",
            noveltyDelta: "0",
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
            newWeights: "",
            newRelevance: 0,
            newDiversity: 0,
            newNovelty: 0,
            middleRelevance: 0.5,
            middleDiversity: 0.5,
            middleNovelty: 0.5
        }
    },
    computed: {
    },
    methods: {
        clipAndRound(x) {
            let result = Number(x.toFixed(2));
            if (result < 0) {
                return 0.0;
            } else if (result > 1) {
                return 1.0;
            }
            return result;
        },
        onRelevanceChange(newRel) {
            var newRelevance = parseFloat(newRel);
            var othersAccum = this.diversity + this.novelty;

            if (othersAccum == 0) {
                return this.clipAndRound(newRelevance);
            }

            var diversityShare = this.diversity / othersAccum;
            var noveltyShare = this.novelty / othersAccum;

            if (newRelevance > this.relevance) {
                // Handle increase
                let diff = newRelevance - this.relevance;
                
                this.diversity -= diversityShare * diff;
                this.novelty -= noveltyShare * diff;

                let totalAccum = newRelevance + this.diversity + this.novelty;
                this.diversity = this.clipAndRound(this.diversity / totalAccum);
                this.novelty = this.clipAndRound(this.novelty / totalAccum);

                return this.clipAndRound(newRelevance / totalAccum); //this.relevance + diff;
            } else if (newRelevance < this.relevance) {
                // Handle decrease
                let diff = this.relevance - newRelevance;
                
                this.diversity += diversityShare * diff;
                this.novelty += noveltyShare * diff;

                let totalAccum = newRelevance + this.diversity + this.novelty;
                this.diversity = this.clipAndRound(this.diversity / totalAccum);
                this.novelty = this.clipAndRound(this.novelty / totalAccum);


                return this.clipAndRound(newRelevance / totalAccum); //this.relevance + diff;
            }

            return this.clipAndRound(newRelevance);
        },
        onDiversityChange(newDiv) {
            var newDiversity = parseFloat(newDiv);
            var othersAccum = this.relevance + this.novelty;

            if (othersAccum == 0) {
                return this.clipAndRound(newDiversity);
            }

            var relevanceShare = this.relevance / othersAccum;
            var noveltyShare = this.novelty / othersAccum;

            if (newDiversity > this.diversity) {
                // Handle increase
                let diff = newDiversity - this.diversity;
                
                this.relevance -= relevanceShare * diff;
                this.novelty -= noveltyShare * diff;

                let totalAccum = newDiversity + this.relevance + this.novelty;
                this.relevance = this.clipAndRound(this.relevance / totalAccum);
                this.novelty = this.clipAndRound(this.novelty / totalAccum);

                return this.clipAndRound(newDiversity / totalAccum);
            } else if (newDiversity < this.diversity) {
                // Handle decrease
                let diff = this.diversity - newDiversity;
                
                this.relevance += relevanceShare * diff;
                this.novelty += noveltyShare * diff;

                let totalAccum = newDiversity + this.relevance + this.novelty;
                this.relevance = this.clipAndRound(this.relevance / totalAccum);
                this.novelty = this.clipAndRound(this.novelty / totalAccum);

                return this.clipAndRound(newDiversity / totalAccum);
            }

            return this.clipAndRound(newDiversity);
        },
        onNoveltyChange(newNov) {
            var newNovelty = parseFloat(newNov);
            var othersAccum = this.relevance + this.diversity;

            if (othersAccum == 0) {
                return this.clipAndRound(newNovelty);
            }

            var relevanceShare = this.relevance / othersAccum;
            var diversityShare = this.diversity / othersAccum;

            if (newNovelty > this.novelty) {
                // Handle increase
                let diff = newNovelty - this.novelty;
                
                this.relevance -= relevanceShare * diff;
                this.diversity -= diversityShare * diff;

                let totalAccum = newNovelty + this.relevance + this.diversity;
                this.relevance = this.clipAndRound(this.relevance / totalAccum);
                this.diversity = this.clipAndRound(this.diversity / totalAccum);

                return this.clipAndRound(newNovelty / totalAccum);
            } else if (newNovelty < this.novelty) {
                // Handle decrease
                let diff = this.novelty - newNovelty;
                
                this.relevance += relevanceShare * diff;
                this.diversity += diversityShare * diff;

                let totalAccum = newNovelty + this.relevance + this.diversity;
                this.relevance = this.clipAndRound(this.relevance / totalAccum);
                this.diversity = this.clipAndRound(this.diversity / totalAccum);

                return this.clipAndRound(newNovelty / totalAccum);
            }

            return this.clipAndRound(newNovelty);
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
            event.preventDefault();
            
            const MAX_SHARE = 0.8; // We can increase at most by this share of the remaining and decrease atmost by this share of current value
            
            let relDelta = parseFloat(this.relevanceDelta);
            let relOfMaxShare = 0.25 * relDelta; 

            let divDelta = parseFloat(this.diversityDelta);
            let divOfMaxShare = 0.25 * divDelta;

            let novDelta = parseFloat(this.noveltyDelta);
            let novOfMaxShare = 0.25 * novDelta;

            let baseRel = relDelta > 0 ? ((1.0 - defaultRelevance) * MAX_SHARE) : defaultRelevance * MAX_SHARE;
            let baseDiv = divDelta > 0 ? ((1.0 - defaultDiversity) * MAX_SHARE) : defaultDiversity * MAX_SHARE;
            let baseNov = novDelta > 0 ? ((1.0 - defaultNovelty) * MAX_SHARE) : defaultNovelty * MAX_SHARE;

            this.newRelevance = defaultRelevance + relOfMaxShare * baseRel;
            this.newDiversity = defaultDiversity + divOfMaxShare * baseDiv;
            this.newNovelty = defaultNovelty + novOfMaxShare * baseNov;

            let sum = this.newRelevance + this.newDiversity + this.newNovelty;

            this.newWeights = `${this.newRelevance/sum},${this.newDiversity/sum},${this.newNovelty/sum}`;
            console.log("New weights are: " + this.newWeights);
            return;
            //this.$forceUpdate();
            this.$nextTick(() => {
                event.target.submit(); 
            });
        },
    },
    async mounted() {
        const btns = document.querySelectorAll(".btn");
        
        // Register the handlers for event reporting
        startViewportChangeReportingWithLimit(`/utils/changed-viewport`, csrfToken, 1.0);
        registerClickedButtonReporting(`/utils/on-input`, csrfToken, btns);
        reportLoadedPage(`/utils/loaded-page`, csrfToken, "refinement_feedback", ()=>
            {
                data = {
                    "result_layout": resultLayout, // TODO REPLACE with layout_version 
                    "initial_weights": {
                        "relevance": defaultRelevance,
                        "diversity": defaultDiversity,
                        "novelty": defaultNovelty
                    }, // TODO add initial weights
                    "processed_inputs": {
                        "relevance": this.newRelevance,
                        "diversity": this.newDiversity,
                        "novelty": this.newNovelty
                    }
                };

                if (resultLayout == "1") {
                    data["raw_inputs"] = { // TODO add slider values and new weights
                        "relevance": this.relevance,
                        "diversity": this.diversity,
                        "novelty": this.novelty
                    }
                }
                else if (resultLayout == "2") {
                    data["raw_inputs"] = {
                        "relevance": this.relevanceValue,
                        "diversity": this.diversityValue,
                        "novelty": this.noveltyValue
                    }
                }
                else if (resultLayout == "3") {
                    data["raw_inputs"] = { // TODO add slider values and new weights
                        "relevance": this.relevanceDelta,
                        "diversity": this.diversityDelta,
                        "novelty": this.noveltyDelta
                    }
                }

                return data;
            }
        );
    }
})