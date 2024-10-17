document.addEventListener('DOMContentLoaded', async () => {
    const loader = document.getElementById('loader');
    const restaurantNameElem = document.getElementById('restaurantName');
    const badgeElem = document.getElementById('badge');
    const summaryElem = document.getElementById('summary');
    const newRatingElem = document.getElementById('newRating');
    const fakeReviewsContainer = document.getElementById('fakeReviewsContainer');
    const realReviewsContainer = document.getElementById('realReviewsContainer');
    const backendURL = 'http://127.0.0.1:5000/analyze-reviews';
    const apiKey = 'AIzaSyBYcVKwQ45LM-tPyUZ0fmjIPrFjcT7OtYs';

    // Tab navigation elements
    const fakeReviewsTab = document.getElementById('fakeReviewsTab');
    const realReviewsTab = document.getElementById('realReviewsTab');

    // Tab switching logic
    fakeReviewsTab.addEventListener('click', () => {
        fakeReviewsTab.classList.add('active');
        realReviewsTab.classList.remove('active');
        fakeReviewsContainer.classList.add('active');
        realReviewsContainer.classList.remove('active');
    });

    realReviewsTab.addEventListener('click', () => {
        realReviewsTab.classList.add('active');
        fakeReviewsTab.classList.remove('active');
        realReviewsContainer.classList.add('active');
        fakeReviewsContainer.classList.remove('active');
    });

    function extractIdentifierFromURL() {
        return new Promise((resolve, reject) => {
            chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
                const url = tabs[0].url;

                if (url.includes('google.com/maps')) {
                    const match = url.match(/\/place\/([^/]+)/) || url.match(/!1s([^!]+)/);

                    if (match && match[1]) {
                        resolve(match[1]);
                    } else {
                        reject('Failed to extract identifier from the URL.');
                    }
                } else {
                    reject('This is not a Google Maps page.');
                }
            });
        });
    }

    async function fetchPlaceDetails(placeIdentifier) {
        loader.style.display = 'block';

        const textSearchURL = `https://maps.googleapis.com/maps/api/place/textsearch/json?query=${encodeURIComponent(placeIdentifier)}&key=${apiKey}`;

        try {
            let response = await fetch(textSearchURL);
            let data = await response.json();

            if (data.results && data.results.length > 0) {
                const placeID = data.results[0].place_id;
                const placeDetailsURL = `https://maps.googleapis.com/maps/api/place/details/json?place_id=${placeID}&key=${apiKey}`;
                let placeDetailsResponse = await fetch(placeDetailsURL);
                let placeDetailsData = await placeDetailsResponse.json();

                if (placeDetailsData.result) {
                    const placeTypes = placeDetailsData.result.types;

                    if (placeTypes.includes('restaurant')) {
                        return {
                            restaurantName: placeDetailsData.result.name || 'Unknown Restaurant',
                            reviews: placeDetailsData.result.reviews ? placeDetailsData.result.reviews.slice(0, 5) : [],
                            isRestaurant: true,
                        };
                    } else {
                        return { isRestaurant: false };
                    }
                } else {
                    return { isRestaurant: false };
                }
            } else {
                return { isRestaurant: false };
            }
        } catch (error) {
            console.error('Failed to fetch place details:', error);
            return { isRestaurant: false };
        } finally {
            loader.style.display = 'none';
        }
    }

    async function sendToBackend(restaurantName, reviews) {
        const reviewsData = reviews.map((review) => ({
            author: review.author_name || 'Unknown Author',
            rating: review.rating || 0,
            review: review.text || '',
            time: review.relative_time_description || ''
        }));

        const payload = {
            restaurant_name: restaurantName,
            reviews: reviewsData
        };

        try {
            loader.style.display = 'block';

            const response = await fetch(backendURL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json();
            displayBackendResults(data);

        } catch (error) {
            console.error('Failed to send data to backend:', error);
        } finally {
            loader.style.display = 'none';
        }
    }

    function displayBackendResults(data) {
        // Badge display based on fake reviews count
        const fakeReviewsCount = data.fake_reviews_detected;

        if (fakeReviewsCount === 0) {
            badgeElem.style.backgroundColor = 'green';
            badgeElem.textContent = "All clear: No fake reviews detected.";
        } else if (fakeReviewsCount >= 1 && fakeReviewsCount <= 2) {
            badgeElem.style.backgroundColor = 'orange';
            badgeElem.textContent = "Warning: Some fake reviews detected. Be cautious.";
        } else {
            badgeElem.style.backgroundColor = 'red';
            badgeElem.textContent = "Alert: Many fake reviews detected! Consumer attention needed.";
        }

        // Summarized review section
        summaryElem.innerHTML = `<strong>Summarized Review:</strong> ${data.summarized_reviews}`;

        // Display new rating with stars
        const starPercentage = (data.new_rating / 5) * 100;
        newRatingElem.innerHTML = `
            <strong>New Rating:</strong> 
            <div class="stars-outer">
                <div class="stars-inner" style="width:${starPercentage}%"></div>
            </div>
            ${data.new_rating} / 5
        `;

        // Fake reviews
        fakeReviewsContainer.innerHTML = '';
        if (data.fake_reviews && data.fake_reviews.length > 0) {
            data.fake_reviews.forEach((review) => {
                const reviewElem = document.createElement('div');
                reviewElem.classList.add('review');
                reviewElem.innerHTML = `
                    <div><strong>Author:</strong> ${review.author}</div>
                    <div><strong>Rating:</strong> ${review.rating} / 5</div>
                    <div><strong>Review:</strong> ${review.review}</div>
                    <div><strong>Time:</strong> ${review.time}</div>
                    <div><strong>Anomalies:</strong> ${review.anomalies.join(', ')}</div>
                `;
                fakeReviewsContainer.appendChild(reviewElem);
            });
        } else {
            fakeReviewsContainer.innerHTML = `<div>No fake reviews detected.</div>`;
        }

        // Real reviews
        realReviewsContainer.innerHTML = '';
        if (data.real_reviews && data.real_reviews.length > 0) {
            data.real_reviews.forEach((review) => {
                const reviewElem = document.createElement('div');
                reviewElem.classList.add('review');
                reviewElem.innerHTML = `
                    <div><strong>Author:</strong> ${review.author}</div>
                    <div><strong>Rating:</strong> ${review.rating} / 5</div>
                    <div><strong>Review:</strong> ${review.review}</div>
                    <div><strong>Time:</strong> ${review.time}</div>
                `;
                realReviewsContainer.appendChild(reviewElem);
            });
        } else {
            realReviewsContainer.innerHTML = `<div>No real reviews found.</div>`;
        }
    }

    try {
        loader.style.display = 'block';

        const placeIdentifier = await extractIdentifierFromURL();
        const placeData = await fetchPlaceDetails(placeIdentifier);

        if (!placeData.isRestaurant) {
            restaurantNameElem.textContent = 'This is not a restaurant.';
        } else {
            restaurantNameElem.textContent = placeData.restaurantName;
            document.title = placeData.restaurantName;

            if (placeData.reviews && placeData.reviews.length > 0) {
                await sendToBackend(placeData.restaurantName, placeData.reviews);
            } else {
                reviewsContainer.textContent = 'No reviews available.';
            }
        }
    } catch (error) {
        console.error(error);
        restaurantNameElem.textContent = 'Error: Could not fetch data.';
    } finally {
        loader.style.display = 'none';
    }
});
