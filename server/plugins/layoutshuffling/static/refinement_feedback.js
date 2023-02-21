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
            newNovelty: 0
        }
    },
    computed: {
    },
    methods: {
        onRelevanceChange(newRel) {
            var newRelevance = parseFloat(newRel);
            var relevance = parseFloat(this.relevance);
            var diversity = parseFloat(this.diversity);
            var novelty = parseFloat(this.novelty);
            var othersAccum = diversity + novelty;

            if (othersAccum == 0) {
                return newRel;
            }

            var diversityShare = diversity / othersAccum;
            var noveltyShare = novelty / othersAccum;

            if (newRelevance > this.relevance) {
                // Handle increase
                let diff = newRelevance - relevance;
                
                diversity -= diversityShare * diff;
                novelty -= noveltyShare * diff;

                let totalAccum = newRelevance + diversity + novelty;
                diversity = (diversity / totalAccum) * 100;
                novelty = (novelty / totalAccum) * 100;

                this.diversity = diversity.toFixed(1);
                this.novelty = novelty.toFixed(1);

                return ((newRelevance / totalAccum) * 100).toFixed(1); //this.relevance + diff;
            } else if (relevance < this.relevance) {
                // Handle decrease
                let diff = relevance - newRelevance;
                
                diversity += diversityShare * diff;
                novelty += noveltyShare * diff;

                let totalAccum = newRelevance + diversity + novelty;
                diversity = (diversity / totalAccum) * 100;
                novelty = (novelty / totalAccum) * 100;

                this.diversity = diversity.toFixed(1);
                this.novelty = novelty.toFixed(1);

                return ((newRelevance / totalAccum) * 100).toFixed(1); //this.relevance + diff;
            }

            return newRel;
        },
        onDiversityChange(newDiv) {
            var newDiversity = parseFloat(newDiv);
            var diversity = parseFloat(this.diversity);
            var relevance = parseFloat(this.relevance);
            var novelty = parseFloat(this.novelty);
            var othersAccum = relevance + novelty;

            if (othersAccum == 0) {
                return newDiv;
            }

            var relevanceShare = relevance / othersAccum;
            var noveltyShare = novelty / othersAccum;

            if (newDiversity > this.diversity) {
                // Handle increase
                let diff = newDiversity - diversity;
                
                relevance -= relevanceShare * diff;
                novelty -= noveltyShare * diff;

                let totalAccum = newDiversity + relevance + novelty;
                relevance = (relevance / totalAccum) * 100;
                novelty = (novelty / totalAccum) * 100;

                this.relevance = relevance.toFixed(1);
                this.novelty = novelty.toFixed(1);

                return ((newDiversity / totalAccum) * 100).toFixed(1);
            } else if (newDiversity < this.diversity) {
                // Handle decrease
                let diff = diversity - newDiversity;
                
                relevance += relevanceShare * diff;
                novelty += noveltyShare * diff;

                let totalAccum = newDiversity + relevance + novelty;
                relevance = (relevance / totalAccum) * 100;
                novelty = (novelty / totalAccum) * 100;

                this.relevance = relevance.toFixed(1);
                this.novelty = novelty.toFixed(1);

                return ((newDiversity / totalAccum) * 100).toFixed(1); 
            }

            return newDiv;
        },
        onNoveltyChange(newNov) {
            var newNovelty = parseFloat(newNov);
            var novelty = parseFloat(this.novelty);
            var relevance = parseFloat(this.relevance);
            var diversity = parseFloat(this.diversity);
            var othersAccum = relevance + diversity;

            if (othersAccum == 0) {
                return newNov;
            }

            var relevanceShare = relevance / othersAccum;
            var diversityShare = diversity / othersAccum;

            if (newNovelty > this.novelty) {
                // Handle increase
                let diff = newNovelty - novelty;
                
                relevance -= relevanceShare * diff;
                diversity -= diversityShare * diff;

                let totalAccum = newNovelty + relevance + diversity;
                relevance = (relevance / totalAccum) * 100;
                diversity = (diversity / totalAccum) * 100;

                this.relevance = relevance.toFixed(1);
                this.diversity = diversity.toFixed(1);

                return ((newNovelty / totalAccum) * 100).toFixed(1);
            } else if (newNovelty < this.novelty) {
                // Handle decrease
                let diff = novelty - newNovelty;
                
                relevance += relevanceShare * diff;
                diversity += diversityShare * diff;

                let totalAccum = newNovelty + relevance + diversity;
                relevance = (relevance / totalAccum) * 100;
                diversity = (diversity / totalAccum) * 100;

                this.relevance = relevance.toFixed(1);
                this.diversity = diversity.toFixed(1);

                return ((newNovelty / totalAccum) * 100).toFixed(1); 
            }

            return newNov;
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
            
            event.target.submit();
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