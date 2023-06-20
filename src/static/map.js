var map = L.map('map').setView([40.631491, -8.656481], 16);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
}).addTo(map);

var obuIcon = L.icon({
    iconUrl: "static/car.png",
    iconSize: [16, 16],
    iconAnchor: [18, 39],
    popupAnchor: [10, -35]
});

//array de markers
var markers = [];

setInterval(obuCall, 1000);

function obuCall() {
    $(document).ready(function(){

        $.ajax({
            url: '',
            type: 'get',
            contentType: 'application/json',
            data: {},
            success: function(response){
                markers.forEach(delMarker)
                
                let i=0;
                for(var key in response){
                    markers[i] = L.marker([ response[key]["lat"], response[key]["long"]], {icon: obuIcon}).addTo(map)
                        .bindTooltip(key, {permanent: false});
                    i++;
                } 
            }
        })

    })
}

function delMarker(value, index, array){
    map.removeLayer(value)
}