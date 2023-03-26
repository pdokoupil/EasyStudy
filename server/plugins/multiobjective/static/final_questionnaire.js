window.app = new Vue({
    el: '#app',
    delimiters: ['[[',']]'], // Used to replace double { escaping with double [ escaping (to prevent jinja vs vue inference)
    data: function() {
        
        return {
            items: [
                {name: "Movie 1", _please_select: {ch1: false, ch2: false, ch3: false}},
                {name: "Movie 2", _please_select: {ch1: false, ch2: false, ch3: false}},
                {name: "Movie 3", _please_select: {ch1: false, ch2: false, ch3: false}}
            ],
            fields: [
                {key: "name", label: "Movie name"},
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