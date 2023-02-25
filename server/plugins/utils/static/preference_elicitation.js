
function preference_elicitation() {

}

var updatedOnce = false;

function elicitation_ctx_lambda() {
    return {
        "items": Array.from(document.getElementsByTagName("img")).map(x => {
            return {
                "id": x.id, // Corresponds to movie idx
                "name": x.name,
                "url": x.src,
                "title": x.title,
                "viewport": getElementBoundingBox(x)
            };
        }),
    };
}

window.app = new Vue({
    el: '#app',
    delimiters: ['[[',']]'], // Used to replace double { escaping with double [ escaping (to prevent jinja vs vue inference)
    data: function() {
    // let impl = document.getElementById("impl").className;
    return {
        userEmail: "{{email}}",
        items: [],
        selected: [],
        impl: impl,
        selectMode: "multi",
        lastSelectedCluster: null,
        handle: false,
        rows: [],
        rows2: [],
        itemsPerRow: 80,
        jumboHeader: "Preference Elicitation",
        disableNextStep: false,
        searchMovieName: null,
        itemsBackup: null,
        rowsBackup: null,
        busy: false
    }
    },
    async mounted() {
        const btns = document.querySelectorAll(".btn");
        
        // This was used for reporting as previously reporting endpoints were defined inside plugin

        // Get the number of items user is supposed to select
        let data = await fetch(initial_data_url + "?impl=" + this.impl + "&i=0").then((resp) => resp.json()).then((resp) => resp);
        
        res = this.prepareTable(data);
        this.rows = res["rows"];
        this.items = res["items"];

        // Register the handlers for event reporting
        startViewportChangeReportingWithLimit(`/utils/changed-viewport`, csrfToken, 1.0, true, elicitation_ctx_lambda);
        registerClickedButtonReporting(`/utils/on-input`, csrfToken, btns, ()=>{
            return {
                "search_text_box_value": this.searchMovieName
            };
        });
        reportLoadedPage(`/utils/loaded-page`, csrfToken, "preference_elicitation", ()=>{
            return {"impl":impl};
        });
        
        setTimeout(function () {
            reportViewportChange(`/utils/changed-viewport`, csrfToken, elicitation_ctx_lambda);
        }, 5000);
    },
    methods: {
        async handlePrefixSearch(movieName) {
            let foundMovies = await fetch(search_item_url + "?attrib=movie&pattern="+movieName).then(resp => resp.json());
            return foundMovies;
        },
        prepareTable(data) {
            let row = [];
            let rows = [];
            let items = [];
            for (var k in data) {
                items.push({
                    "movieName": data[k]["movie"],
                    "movie": {
                        "idx": data[k]["movie_idx"],
                        "url": data[k]["url"]
                    }
                });
                row.push({
                    "movieName": data[k]["movie"],
                    "movie": {
                        "idx": data[k]["movie_idx"],
                        "url": data[k]["url"]
                    }
                });
                if (row.length >= this.itemsPerRow) {
                    rows.push(row);
                    row = [];
                }
            }
            if (row.length > 0) {
                rows.push(row);
            }

            return {"rows": rows, "items": items };
        },
        async onClickSearch(event) {
            reportOnInput("/utils/on-input", csrfToken, "search", {"search_text_box_value": this.searchMovieName});
            let data = await this.handlePrefixSearch(this.searchMovieName);
            let res = this.prepareTable(data);
            
            // Do not overwrite backups when doing repeated search
            if (this.itemsBackup === null) {
                this.itemsBackup = this.items;
                this.rowsBackup = this.rows;
            }

            this.rows = res["rows"];
            this.items = res["items"];

            // VUE is reusing dom and dropping classes, add them back
            this.$nextTick(() => {
                for (let i in this.selected) {
                    let el = document.getElementById(this.selected[i].movie.idx);
                    if (el) {
                        el.classList.remove("selected");
                        el.classList.add("selected");
                    }
                }
            });
        },
        onKeyDownSearchMovieName(e) {
            if (e.key === "Enter") {
                this.onClickSearch(null);
            }
        },
        onClickCancelSearch() {
            this.items = this.itemsBackup;
            this.rows = this.rowsBackup;
            this.itemsBackup = null;
            this.rowsBackup = null;

            // VUE is reusing dom and dropping classes, add them back
            this.$nextTick(() => {
                for (let i in this.selected) {
                    let el = document.getElementById(this.selected[i].movie.idx);
                    if (el) {
                        el.classList.remove("selected");
                        el.classList.add("selected");
                    }
                }
            });
        },
        async onClickLoadMore() {
            let data = await fetch(initial_data_url + "?impl=" + this.impl).then((resp) => resp.json()).then((resp) => resp);
            res = this.prepareTable(data);
            this.rows = res["rows"];
            this.items = res["items"];

            // VUE is reusing dom and dropping classes, add them back
            this.$nextTick(() => {
                for (let i in this.selected) {
                    let el = document.getElementById(this.selected[i].movie.idx);
                    if (el) {
                        el.classList.remove("selected");
                        el.classList.add("selected");
                    }
                }
            });
        },
        onUpdateSearchMovieName(newValue) {
        },
        movieIndexOf(arr, item) {
            for (let idx in arr) {
                let arrItem = arr[idx];
                if (arrItem.movie.idx === item.movie.idx
                    && arrItem.movieName === item.movieName
                    && arrItem.movie.url === item.movie.url) {
                        return idx;
                    }
            }
            return -1;
        },
        onSelectMovie(event, item) {
            // TODO wrap movieIndexOf as generic indexOf with selector lambda
            let index = this.movieIndexOf(this.selected, item); //this.selected.indexOf(item);
            if (index > -1) {
                // Already there, remove it
                this.selected.splice(index, 1);
                event.srcElement.classList.remove("selected");
                reportDeselectedItem(`/utils/deselected-item`, csrfToken, item, this.selected);
            } else {
                // Not there, insert
                this.selected.push(item);
                event.srcElement.classList.add("selected");
                reportSelectedItem(`/utils/selected-item`, csrfToken, item, this.selected);
            }
        },
        onRowClicked(item) {
            let index = this.movieIndexOf(this.selected, item); // this.selected.indexOf(item);
            if (index > -1) {
                this.selected.splice(index, 1);
                //this.$refs.selectableTable.unselectRow(this.items.indexOf(item));
            } else {
                this.selected.push(item);
                //this.$refs.selectableTable.selectRow(this.items.indexOf(item));
            }
        },
        onElicitationFinish(form) {
            this.busy = true;
            let selectedMoviesTag = document.createElement("input");
            selectedMoviesTag.setAttribute("type","hidden");
            selectedMoviesTag.setAttribute("name","selectedMovies");
            selectedMoviesTag.setAttribute("value", this.selected.map((x) => x.movie.idx).join(","));

            form.appendChild(selectedMoviesTag);

            form.submit();
        }
    }
})