function reportMessage(endpoint, messageType, csrfToken, sourceLocation, message, extraCtxLambda=()=>"") {
    data = {
        "message": message,
        "type": messageType,
        "source": sourceLocation,
        "time": new Date().toISOString(),
        "extra": extraCtxLambda()
    };
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

function reportError(endpoint, csrfToken, sourceLocation, errorMessage, extraCtxLambda=()=>"") {
    reportMessage(endpoint, "error", csrfToken, sourceLocation, errorMessage, extraCtxLambda);
}