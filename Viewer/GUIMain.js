/**
 * The main Javascript code for setting up tabs and gluing all of the interaction
 * and components together, as well as handling file loading
 */


/** Main audio object */
function AudioObject() {
    this.audio_widget = document.getElementById("audio_widget");
    this.times = [0];
    this.dim = 0;
    
    /**
     * Return the index of the row with the closest time to
     * the currently playing time in the audio widget
     */
    this.getClosestIdx = function() {
        //TODO: Make a binary search
        var time = audio_widget.currentTime;
        var t = 0;
		var mindiff = Math.abs(time - this.times[0]);
		for (var i = 1; i < this.times.length; i++) {
		    var diff = Math.abs(this.times[i] - time);
		    if (diff < mindiff) {
		        mindiff = diff;
		        t = i;
		    }
		}
		return t;
    }
}
var audio_obj = new AudioObject();
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


function setupSong(params) {
    //Setup audio buffers
    audio_obj.audio_widget.src = params.audio;
    audio_obj.times = params.times;
    audio_obj.dim = params.dim;
    songnameTxt.innerHTML = params.songname;
    for (var i = 0; i < tabdata.length; i++) {
        tabdata[i].canvas.updateParams(params);
    }
    refreshDisplays();
    progressBar.changeToReady();
}

/** Setup file input button handler */
var fileInput = document.getElementById('fileInput');
fileInput.addEventListener('change', function(e) {
    var file = fileInput.files[0];
    var reader = new FileReader();
    reader.onload = function(e) {
        var params = JSON.parse(reader.result);
        setupSong(params);
    };
    reader.readAsText(file);
});


function loadPrecomputedSong(file) {
    progressBar.loadString = "Reading data from server";
    progressBar.loadColor = "red";
    progressBar.loading = true;
    progressBar.changeLoad();

    var xhr = new XMLHttpRequest();
    xhr.open('GET', file, true);
    xhr.responseType = 'json';
    xhr.onload = function(err) {
        setupSong(this.response);
    };
    progressBar.loading = true;
    progressBar.ndots = 0;
    progressBar.changeLoad();
    xhr.send();

}

audio_obj.audio_widget.addEventListener("play", refreshDisplays);
audio_obj.audio_widget.addEventListener("pause", refreshDisplays);
audio_obj.audio_widget.addEventListener("seek", refreshDisplays);
