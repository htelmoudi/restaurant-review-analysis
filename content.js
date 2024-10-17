function extractIdentifierFromURL() {
    let identifier = null;

    // Extract from a typical Google Maps place URL
    const url = window.location.href;

    // Attempt to match a pattern after /place/ or after !1s for business identifiers
    const match = url.match(/\/place\/([^/]+)/) || url.match(/!1s([^!]+)/);

    if (match && match[1]) {
        identifier = match[1];
    }

    if (identifier) {
        // Store the identifier in Chrome's local storage for further use
        chrome.storage.local.set({ placeIdentifier: identifier });
        console.log(`Extracted Identifier: ${identifier}`);
    } else {
        console.error('Failed to extract identifier from the URL.');
    }
}

// Run the function to extract the identifier
extractIdentifierFromURL();
