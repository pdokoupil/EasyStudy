
function validateScreenResolution(minW, minH) {
    let success = true;

    if (minW) {
        success = success && window.innerWidth >= minW;
    }

    if (minH) {
        success = success && window.innerHeight >= minH;
    }

    return success;
}

function getScreenSizes() {
    return {
        "window.screen.height": window.screen.height,
        "document.body.scrollHeight": document.body.scrollHeight,
        "window.innerHeight": window.innerHeight,
        "window.screen.availHeight": window.screen.availHeight,
        "document.body.clientHeight": document.body.clientHeight,
        "window.screen.width": window.screen.width,
        "document.body.scrollWidth": document.body.scrollWidth,
        "window.innerWidth": window.innerWidth,
        "window.screen.availWidth": window.screen.availWidth,
        "document.body.clientWidth": document.body.clientWidth
    };
}