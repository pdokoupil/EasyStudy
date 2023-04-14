Vue.config.errorHandler = function(err, vm, info) {
    let err_json = JSON.stringify(err, Object.getOwnPropertyNames(err));
    reportError("/utils/on-message", csrfToken, window.location.href, err, ()=> {
        return {
            "info": info,
            "src_handler": "vue",
            "err_json": err_json
        }
    });
    return true;
}

window.app = new Vue({
    el: '#app',
    delimiters: ['[[',']]'], // Used to replace double { escaping with double [ escaping (to prevent jinja vs vue inference)
    data: function() {
        
        
        let items = [];

        for (let i in attentionMovies) {
            let row = attentionMovies[i];
            items.push({
                movie: {
                    name: row["movie"],
                    movie_idx: row["movie_idx"],
                    url: row["url"]
                },
                _please_select: {ch: null}
            });
        }


        return {
            items: items,
            fields: [
                {key: "movie", label: "Movie"},
                {key: "_please_select", label: ""}
            ]
        }
    },
    computed: {
    },
    methods: {
        
    },
    async mounted() {
        const btns = document.querySelectorAll(".btn");
        const radios = document.querySelectorAll("input[type=radio]");
        
        // Register the handlers for event reporting
        startViewportChangeReportingWithLimit(`/utils/changed-viewport`, csrfToken, 1.0, true);
        startScrollReportingWithLimit(`/utils/changed-viewport`, csrfToken, 1.0, document.getElementsByName("scrollableDiv"));
        registerClickedButtonReporting(`/utils/on-input`, csrfToken, btns);
        registerClickedRadioReporting("/utils/on-input", csrfToken, radios);
        reportLoadedPage(`/utils/loaded-page`, csrfToken, "final_questionnaire");
    }
})