//Programmer: Chris Tralie

/** A canvas for drawing a self-similarity matrix synchronized with audio */

/**
 * Create a structure canvas
 * @param {object} audio_obj - A dictionary of audio parameters, including 
 * 	a handle to the audio widget, an array of times of each window,
 *  and an array of the current index into the time array
 */
function StructureCanvas(audio_obj) {
	this.ssmcanvas = d3.select("#SimilarityCanvas");
    this.eigcanvas = d3.select("#EigCanvas");
	this.CSImage = this.ssmcanvas.append('image');
	this.ssmlinehoriz = this.ssmcanvas.append('line')
			.attr('x1', 0).attr('x2', 800)
			.attr('y1', 0).attr('y2', 0)
			.style('fill', 'cyan');
	this.ssmlinevert = this.ssmcanvas.append('line')
			.attr('x1', 0).attr('x2', 0)
			.attr('y1', 0).attr('y2', 800)
			.style('fill', 'cyan');
	this.EigImage = this.eigcanvas.append('image');
	this.eiglinevert = this.eigcanvas.append('line')
			.attr('x1', 0).attr('x2', 0)
			.attr('y1', 0).attr('y2', 10)
			.style('fill', 'cyan');
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
		if ('v' in params) {
			this.EigImage.attr('href', params.v)
						 .attr('width', params.dim)
						 .attr('height', params.v_height);
			this.eigcanvas.attr('width', params.dim)
						  .attr('height', params.v_height);
		}
		this.ssmcanvas.attr('width', params.dim)
					  .attr('height', params.dim);
		requestAnimationFrame(this.repaint.bind(this));
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
			this.audio_obj.offsetidx = offset1idx;
		}
		else {
			this.audio_obj.offsetidx = offset2idx;
		}
		this.audio_obj.audio_widget.currentTime = this.audio_obj.times[this.audio_obj.offsetidx];
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
		this.audio_obj.offsetidx = evt.offsetX;
		this.audio_obj.audio_widget.currentTime = this.audio_obj.times[this.audio_obj.offsetidx];
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
	this.drawCanvas = function() {
		this.ssmlinehoriz.attr('y1', this.audio_obj.offsetidx)
						 .attr('y2', this.audio_obj.offsetidx);
		this.ssmlinevert.attr('x1', this.audio_obj.offsetidx)
						 .attr('x2', this.audio_obj.offsetidx);
		this.eiglinevert.attr('x1', this.audio_obj.offsetidx)
						 .attr('x2', this.audio_obj.offsetidx);
	};
	
	/**
	 * A function that should be called in conjunction with requestionAnimationFrame
	 * to refresh this canvas.  Continually generates callbacks as long as the
	 * audio is playing, but stops generating callbacks when it is paused to save
	 * computation
	 */
	this.repaint = function() {
		var t = this.audio_obj.audio_widget.currentTime;
		while (this.audio_obj.times[this.audio_obj.offsetidx] < t && 
				this.audio_obj.offsetidx < this.audio_obj.times.length - 1) {
			this.audio_obj.offsetidx++;
		}
		this.drawCanvas();
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
		canvas = document.getElementById('SimilarityCanvas');
		this.initDummyListeners(canvas);
		canvas.addEventListener('mouseup', this.releaseClickEig.bind(this));
		canvas.addEventListener('touchend', this.releaseClickEig.bind(this));
	};

	this.initCanvasHandlers();
}