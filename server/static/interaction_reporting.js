// This script registers some of the handlers that are commonly used for interaction reporting and makes sure they are delivered to the server

// Returns bounding box of given element w.r.t. viewport
function getElementBoundingBox(element) {
    let x = element.getBoundingClientRect();
    return {
        "left": x.left,
        "top": x.top,
        "width": x.width,
        "height": x.height
    };
}

function getViewportBoundingBox() {
    return getElementBoundingBox(document.documentElement);
}

// Returns context, something that is relevant for most of the reported status and that makes
// sense to accompany reported information
function getContext(extra="") {
    return {
        "url": window.location.href,
        "time": new Date().toISOString(),
        "viewport": getViewportBoundingBox(),
        "extra": extra
    };
}

function reportViewportChange(endpoint, csrfToken, extraCtxLambda=()=>"") {
    data = {
        "viewport": getViewportBoundingBox(),
        "screen_sizes": getScreenSizes(),
        "context": getContext(extraCtxLambda())
    }
    return fetch(endpoint,
        {
            method: "POST",
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken
            },
            body: JSON.stringify(data),
            redirect: "follow"
        }
    )
}

// Starts listening for viewport changes and posts them to the given endpoint
// initialReport parameter allow us to report about current values
// (this is especially useful if we want to report initial viewport dimensions, before any user action)
function startViewportChangeReporting(endpoint, csrfToken, initialReport=true, extraCtxLambda=()=>"") {
    
    if (initialReport === true) {
        window.addEventListener("load", function(e) {
            reportViewportChange(endpoint, csrfToken, extraCtxLambda);
        });
    }

    window.addEventListener("scroll", function(e) {
        reportViewportChange(endpoint, csrfToken, extraCtxLambda);
    });

    window.addEventListener("resize", function(e) {
        reportViewportChange(endpoint, csrfToken, extraCtxLambda);
    });
}


// Starts listening for viewport changes and posts them to the given endpoint
// initialReport parameter allow us to report about current values
// (this is especially useful if we want to report initial viewport dimensions, before any user action)
function startViewportChangeReportingWithLimit(endpoint, csrfToken, timeLimitSeconds, initialReport=true, extraCtxLambda=()=>"") {
    if (initialReport === true) {
        window.addEventListener("load", function(e) {
            reportViewportChange(endpoint, csrfToken, extraCtxLambda);
        });
    }

    var lastReported = new Date();
    var lastReportedViaResie = new Date();

    window.addEventListener("scroll", function(e) {
        let now = new Date();
        if ((now - lastReported) / 1000 > timeLimitSeconds) {
            reportViewportChange(endpoint, csrfToken, extraCtxLambda);
            lastReported = now;
        }
    });

    window.addEventListener("resize", function(e) {
        let now = new Date();
        if ((now - lastReportedViaResie) / 1000 > timeLimitSeconds) {
            reportViewportChange(endpoint, csrfToken, extraCtxLambda);
            lastReportedViaResie = now;
        }
    });
}

// Starts listening for scrolling and posts them to the given endpoint
// initialReport parameter allow us to report about current values
// (this is especially useful if we want to report initial viewport dimensions, before any user action)
// This overloads also accepts elements on which we listen for scroll events
function startScrollReportingWithLimit(endpoint, csrfToken, timeLimitSeconds, elements, extraCtxLambda=()=>"") {
    var lastReported = new Date();
    for (let i = 0; i < elements.length; ++i) {
        elements[i].addEventListener("scroll", function(e) {
            let now = new Date();
            if ((now - lastReported) / 1000 > timeLimitSeconds) {
                reportViewportChange(endpoint, csrfToken, extraCtxLambda);
                lastReported = now;
            }
        });
    }
}

// Used for reporting clicked buttons, options, checkboxes, ratings, etc.
function reportOnInput(endpoint, csrfToken, inputType, data, extraCtxLambda=()=>"") {
    data["context"] = getContext(extraCtxLambda());
    data["input_type"] = inputType;
    fetch(endpoint,
        {
            method: "POST",
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken
            },
            body: JSON.stringify(data),
            redirect: "follow"
        }
    );
}

function registerClickedButtonReporting(endpoint, csrfToken, btns, extraCtxLambda=()=>"") { 
    btns.forEach(btn => {
        btn.addEventListener('click', event => {
            let data = {
                "id": event.target.id,
                "text": event.target.textContent,
                "name": event.target.name
            };
            reportOnInput(endpoint, csrfToken, "button", data, extraCtxLambda);
        });
    });
}

function registerClickedCheckboxReporting(endpoint, csrfToken, checkboxes, extraCtxLambda=()=>"") {
    checkboxes.forEach(checkbox => {
        checkbox.addEventListener('click', event => {
            let data = {
                "id": event.target.id,
                "checked": event.target.checked,
                "name": event.target.name
            };
            reportOnInput(endpoint, csrfToken, "checkbox", data, extraCtxLambda);
        });
    });
}

function registerClickedRadioReporting(endpoint, csrfToken, radios, extraCtxLambda=()=>"") {
    radios.forEach(radio => {
        radio.addEventListener('click', event => {
            let data = {
                "id": event.target.id,
                "value": event.target.value,
                "name": event.target.name
            };
            reportOnInput(endpoint, csrfToken, "radio", data, extraCtxLambda);
        });
    });
}

function reportLoadedPage(endpoint, csrfToken, pageName, extraCtxLambda=()=>"") {
    data = {
        "page": pageName,
        "context": getContext(extraCtxLambda())
    };
    return fetch(endpoint,
        {
            method: "POST",
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken
            },
            body: JSON.stringify(data),
            redirect: "follow"
        }
    )
}

function reportSelectedItem(endpoint, csrfToken, selectedItem, selectedItems, extraCtxLambda=()=>"") {
    data = {
        "selected_item": selectedItem,
        "selected_items": selectedItems,
        "context": getContext(extraCtxLambda())
    };
    return fetch(endpoint,
        {
            method: "POST",
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken
            },
            body: JSON.stringify(data),
            redirect: "follow"
        }
    )
}

function reportDeselectedItem(endpoint, csrfToken, deselectedItem, selectedItems, extraCtxLambda=()=>"") {
    data = {
        "deselected_item": deselectedItem,
        "selected_items": selectedItems,
        "context": getContext(extraCtxLambda())
    };
    return fetch(endpoint,
        {
            method: "POST",
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken
            },
            body: JSON.stringify(data),
            redirect: "follow"
        }
    )
}