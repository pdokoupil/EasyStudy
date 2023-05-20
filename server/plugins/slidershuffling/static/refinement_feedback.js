window.app = new Vue({
    el: '#app',
    delimiters: ['[[',']]'], // Used to replace double { escaping with double [ escaping (to prevent jinja vs vue inference)
    data: function() {
        
        return {
            
        }
    },
    computed: {
    },
    methods: {
        
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