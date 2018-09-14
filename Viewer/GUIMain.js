/**
 * The main Javascript code for setting up tabs and gluing all of the interaction
 * and components together, as well as handling file loading
 */


/** Main audio object */
var audio_obj = {'audio_widget':document.getElementById("audio_widget"), 
                            'time_interval':0, 'dim':0};
audio_obj.audio_widget.style.width = 800;
/** Objects for tab data {name, canvas object, active} */
var tabdata = [{name:"Self-Similarity", canvas:new SimilarityCanvas(audio_obj), active:false}, 
               {name:"Force Graph", canvas:new ForceCanvas(audio_obj), active:false},
               {name:"3D Diffusion Maps", canvas:new DiffusionGLCanvas(audio_obj), active:false}];


/** Setup a handler to refresh the appropriate tab whenever something changes */
refreshDisplays = function() {
    for (i = 0; i < tabdata.length; i++) {
        if (tabdata[i].active) {
            requestAnimationFrame(tabdata[i].canvas.repaint.bind(tabdata[i].canvas));
        }
    }
};

/** Additional display variables */
progressBar = new ProgressBar();
var songnameTxt = document.getElementById("songname");

/** Setup tabs and initialize canavses */
changeTab = function(tab, i) {
    d3.select('.tabmenu').selectAll('button')
        .attr("class", function(d) {
        // Change display style of active tab relative to others
        if (d == tab) {
            d.active = true;
            return "active";
        }
        d.active = false;
        return "";
        })
        .each(function(d) {
        // Hide inactive tabs
        if (d == tab) {
            d.canvas.show();
        }
        else {
            d.canvas.hide();
        }
        });
        refreshDisplays();
};
d3.select('.tabmenu').selectAll("button")
                    .data(tabdata)
                    .enter().append("button").on('click', changeTab)
                    .text(function (d) {return d.name;})
                    .attr('class', function(d, i) {
                        if (i == 0) {
                            return "active";
                        }
                        return "";
                    });
// By default, show only the first canvas
for (var i = 0; i < tabdata.length; i++) {
    if (i == 0) {
        tabdata[i].active = true;
        tabdata[i].canvas.show();
    }
    else {
        tabdata[i].active = false;
        tabdata[i].canvas.hide();
    }
}


/** Setup file input button handler */
var fileInput = document.getElementById('fileInput');
fileInput.addEventListener('change', function(e) {
    var file = fileInput.files[0];
    var reader = new FileReader();
    reader.onload = function(e) {
        var params = JSON.parse(reader.result);
        //Setup audio buffers
        audio_obj.audio_widget.src = params.audio;
        audio_obj.time_interval = params.time_interval;
        audio_obj.dim = params.dim;
        songnameTxt.innerHTML = params.songname;
        for (var i = 0; i < tabdata.length; i++) {
            tabdata[i].canvas.updateParams(params);
        }
        refreshDisplays();
        progressBar.changeToReady();
    };
    reader.readAsText(file);
});


audio_obj.audio_widget.addEventListener("play", refreshDisplays);
audio_obj.audio_widget.addEventListener("pause", refreshDisplays);
audio_obj.audio_widget.addEventListener("seek", refreshDisplays);