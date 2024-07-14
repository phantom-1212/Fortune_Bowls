let map;

function initMap() {
    const bengaluru = { lat: 12.971599, lng: 77.594566 };
    map = new google.maps.Map(document.getElementById("map"), {
        center: bengaluru,
        zoom: 12,
    });

    // Add a marker at Bengaluru
    const marker = new google.maps.Marker({
        position: bengaluru,
        map: map,
        title: "Bengaluru, Karnataka, India",
        draggable: true, // Allow the marker to be dragged
    });

    // Add an event listener to update coordinates when marker is dragged
    google.maps.event.addListener(marker, "dragend", function (event) {
        document.getElementById("latitude").innerText = event.latLng.lat().toFixed(6);
        document.getElementById("longitude").innerText = event.latLng.lng().toFixed(6);
    });
}
