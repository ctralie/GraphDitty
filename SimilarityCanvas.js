//Programmer: Chris Tralie

/** A canvas for drawing a self-similarity matrix synchronized with audio */

/**
 * Create a self-similarity canvas
 * @param {object} audio_obj - A dictionary of audio parameters, including 
 * 	a handle to the audio widget and a time interval between adjacent rows
 *  of the SSM, and the dimension "dim" of the SSM
 */
function SimilarityCanvas(audio_obj) {
	this.ssmcanvas = d3.select("#SimilarityCanvas");
    this.eigcanvas = d3.select("#EigCanvas");
	this.CSImage = this.ssmcanvas.append('image');
	this.ssmlinehoriz = this.ssmcanvas.append('line')
			.attr('x1', 0).attr('x2', 0)
			.attr('y1', 0).attr('y2', 0)
			.attr('stroke-width', 2)
			.attr('stroke', 'cyan');
	this.ssmlinevert = this.ssmcanvas.append('line')
			.attr('x1', 0).attr('x2', 0)
			.attr('y1', 0).attr('y2', 0)
			.style('fill', 'cyan')
			.attr('stroke-width', 2)
			.attr('stroke', 'cyan');
	this.EigImage = this.eigcanvas.append('image');
	this.eiglinevert = this.eigcanvas.append('line')
			.attr('x1', 0).attr('x2', 0)
			.attr('y1', 0).attr('y2', 0)
			.attr('stroke-width', 2)
			.attr('stroke', 'cyan');
	this.audio_obj = audio_obj;

	/**
	 * A function use to update the images on the canvas and to
	 * resize things accordingly
	 * @param {object} params : A dictionary of parameters returned
	 * from the Python program as a JSON file
	 */
	this.updateParams = function(params) {
		this.CSImage.attr('href', params.W)
					.attr('width', params.dim)
					.attr('height', params.dim);
		this.audio_obj.audio_widget.style.width = params.dim;
		this.ssmlinehoriz.attr('x2', params.dim);
		this.ssmlinevert.attr('y2', params.dim);
		if ('v' in params) {
			this.EigImage.attr('href', params.v)
						 .attr('width', params.dim)
						 .attr('height', params.v_height);
			this.eigcanvas.attr('width', params.dim)
						  .attr('height', params.v_height);
			this.eiglinevert.attr('y2', params.dim);
		}
		this.ssmcanvas.attr('width', params.dim)
					  .attr('height', params.dim);
	};

	/**
	 * A function which toggles all of the visible elements to show
	 */
	this.show = function() {
		this.ssmcanvas.style("display", "block");
		this.eigcanvas.style("display", "block");
	};

	/**
	 * A function which toggles all of the visible elements to hide
	 */
	this.hide = function() {
		this.ssmcanvas.style("display", "none");
		this.eigcanvas.style("display", "none");
	};

	/**
	 * A click release handler that is used to seek through the self-similarity 
	 * canvas and to seek to the corresponding position in audio.
	 * Left click seeks to the row and right click seeks to the column
	 * @param {*} evt: Event handler 
	 */
	this.releaseClickSSM = function(evt) {
		evt.preventDefault();
		var offset1idx = evt.offsetY;
		var offset2idx = evt.offsetX;
		var offsetidx = 0;
	
		clickType = "LEFT";
		evt.preventDefault();
		if (evt.which) {
			if (evt.which == 3) clickType = "RIGHT";
			if (evt.which == 2) clickType = "MIDDLE";
		}
		else if (evt.button) {
			if (evt.button == 2) clickType = "RIGHT";
			if (evt.button == 4) clickType = "MIDDLE";
		}
		if (clickType == "LEFT") {
			offsetidx = offset1idx;
		}
		else {
			offsetidx = offset2idx;
		}
		this.audio_obj.audio_widget.currentTime = this.audio_obj.time_interval*offsetidx;
		this.repaint();
		return false;
	};
	
	/**
	 * A click release handler that is used to seek through the Laplacian eigenvector
	 * canvas and to seek to the corresponding position in audio
	 * @param {*} evt: Event handler 
	 */
	this.releaseClickEig = function(evt) {
		evt.preventDefault();
		this.audio_obj.audio_widget.currentTime = evt.offsetX*this.audio_obj.time_interval;
		this.repaint();
		return false;
	};
	
	/**
	 * An event handler that does nothing
	 * @param {*} evt 
	 */
	this.dummyHandler = function(evt) {
		evt.preventDefault();
		return false;
	};
	/**
	 * A fuction which renders the SSM and laplacian eigenvectors
	 * with lines superimposed to show where the audio is
	 */
	this.updateCanvas = function() {
		if (this.audio_obj.time_interval > 0) {
			var t = this.audio_obj.audio_widget.currentTime / this.audio_obj.time_interval;
			this.ssmlinehoriz.attr('y1', t).attr('y2', t);
			this.ssmlinevert.attr('x1', t).attr('x2', t);
			this.eiglinevert.attr('x1', t).attr('x2', t);
		}
	};
	
	/**
	 * A function that should be called in conjunction with requestionAnimationFrame
	 * to refresh this canvas so that it is properly synchronized with the audio as it plays.  
	 * It continually generates callbacks as long as the audio is playing, but stops generating 
	 * callbacks when it is paused, to save computation.
	 */
	this.repaint = function() {
		this.updateCanvas();
		if (!this.audio_obj.audio_widget.paused) {
			requestAnimationFrame(this.repaint.bind(this));
		}
	};

	this.initDummyListeners = function(canvas) {
		canvas.addEventListener("contextmenu", function(e){ e.stopPropagation(); e.preventDefault(); return false; }); //Need this to disable the menu that pops up on right clicking
		canvas.addEventListener('mousedown', this.dummyHandler.bind(this));
		canvas.addEventListener('mousemove', this.dummyHandler.bind(this));
		canvas.addEventListener('touchstart', this.dummyHandler.bind(this));
		canvas.addEventListener('touchmove', this.dummyHandler.bind(this));
		canvas.addEventListener('contextmenu', function dummy(e) { return false });
	};
	
	this.initCanvasHandlers = function() {
		var canvas = document.getElementById('SimilarityCanvas');
		this.initDummyListeners(canvas);
		canvas.addEventListener('mouseup', this.releaseClickSSM.bind(this));
		canvas.addEventListener('touchend', this.releaseClickSSM.bind(this));
		canvas = document.getElementById('EigCanvas');
		this.initDummyListeners(canvas);
		canvas.addEventListener('mouseup', this.releaseClickEig.bind(this));
		canvas.addEventListener('touchend', this.releaseClickEig.bind(this));
	};

	this.initCanvasHandlers();
}