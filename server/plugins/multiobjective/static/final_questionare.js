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
        const radios = document.querySelectorAll("input[type=radio]");
        
        // Register the handlers for event reporting
        startViewportChangeReportingWithLimit(`/utils/changed-viewport`, csrfToken, 1.0, true);
        startScrollReportingWithLimit(`/utils/changed-viewport`, csrfToken, 1.0, document.getElementsByName("scrollableDiv"));
        registerClickedButtonReporting(`/utils/on-input`, csrfToken, btns);
        registerClickedRadioReporting("/utils/on-input", csrfToken, radios);
        reportLoadedPage(`/utils/loaded-page`, csrfToken, "final_questionare");
    }
})